import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import glob
import re
from tqdm import tqdm
from datetime import datetime, timedelta
import csv
from collections import defaultdict 
from rasterio.features import rasterize
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform
from pyproj import Transformer
from rasterio.mask import mask
from skimage.feature import graycomatrix, graycoprops
from skimage import io
import numpy as np
from scipy.spatial.distance import cdist

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(dirname))

from parse_args import create_parser
parser = create_parser(mode='test')
test_config = parser.parse_args()
test_config.pid = os.getpid()


from src.model_utils import get_model, load_checkpoint
from src import utils
from s2cloudless import S2PixelCloudDetector

def read_tif(path_IMG):
    with rasterio.open(path_IMG) as tif:
        return tif, tif.read().astype(np.float32)

def process_MS(img):
    intensity_min, intensity_max = 0, 10000
    img = np.clip(img, intensity_min, intensity_max)
    img = (img - intensity_min) / (intensity_max - intensity_min)
    return np.nan_to_num(img)

def process_SAR(img):
    dB_min, dB_max = -25, 0
    img = np.clip(img, dB_min, dB_max)
    img = (img - dB_min) / (dB_max - dB_min)
    return np.nan_to_num(img)

def get_cloud_map(img, cloud_detector):
    """Generates a cloud mask from a raw S2 image array."""
    # The s2cloudless library expects image values in the original 0-10000 range
    # and the channel dimension to be last.
    img_for_detector = np.clip(img, 0, 10000)
    img_for_detector = np.moveaxis(img_for_detector/10000, 0, -1)
    cloud_mask = np.ones((img.shape[-1], img.shape[-1]))
    
    cloud_mask = cloud_detector.get_cloud_masks(img_for_detector[None, ...])[0, ...]
    return cloud_mask.astype(np.float32)

def save_geotiff(path, array, source_tif_meta):
    """Saves a numpy array as a georeferenced TIFF file."""
    metadata = source_tif_meta.copy()
    # Update metadata for the 13-band S2 output
    metadata.update({
        'dtype': array.dtype.name,
        'count': array.shape[0]
    })
    with rasterio.open(path, 'w', **metadata) as dst:
        dst.write(array)


# def get_date_from_filename(filepath):
#     filename = os.path.basename(filepath)
#     match = re.search(r'_(\d{8}T\d{6})_', filename)
#     if match: return datetime.strptime(match.group(1), '%Y%m%dT%H%M%S')
#     return None

def get_date_from_filename(filepath):
    """
    Extracts a date from a filename by trying multiple common date formats.
    """
    filename = os.path.basename(filepath)
    
    date_formats = [
        '%Y%m%dT%H%M%S',  # e.g., 20240110T050159 (Sentinel format)
        '%Y%m%d',         # e.g., 20240110
        '%Y-%m-%d',       # e.g., 2024-01-10
        '%d-%m-%Y',       # e.g., 10-01-2024
        '%m-%d-%Y'        # e.g., 01-10-2024
    ]
    
    # Regex to find any potential date-like strings in the filename
    potential_dates = re.findall(r'\d{8}T\d{6}|\d{8}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}', filename)
    
    for date_str in potential_dates:
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
    return None

def create_cluster_field_mapping(root_directory, field_data_file):
    """
    Reads a CSV or Excel file and groups fields by cluster ID, extracting 
    field_id, geo_json, and the four specified date fields.
    """
    # field_data_file = os.path.join(root_directory, field_data_file_name)
    if not os.path.isfile(field_data_file):
        print(f"Error: Data file not found at {field_data_file}")
        return {}

    try:
        file_extension = os.path.splitext(field_data_file)[1]
        if file_extension == '.csv':
            df = pd.read_csv(field_data_file)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(field_data_file)
        else:
            print(f"Error: Unsupported file type '{file_extension}'. Please use .csv or .xlsx.")
            return {}
            
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return {}

    mapping = defaultdict(list)
    
    for index, row in df.iterrows():
        cluster_id = row.get('cluster_label')
        
        if pd.notna(cluster_id):
            try:
                field_info = {
                    'field_id': row.get('field_id'),
                    'geo_json': eval(row.get('geo_json')),
                    'image_start_date': row.get('image_start_date'),
                    'image_end_date': row.get('image_end_date'),
                    'tile_image_start_date': row.get('tile_image_start_date'),
                    'tile_image_end_date': row.get('tile_image_end_date')
                }
                mapping[cluster_id].append(field_info)
            except (json.JSONDecodeError, TypeError):
                print(f"Warning: Could not parse geo_json for field_id {row.get('field_id')}. Skipping.")

    return dict(mapping)

def find_input_pairs(root_directory):
    s2_base_path = os.path.join(root_directory, 'sentinel-2', 'level-1c')
    s1_base_path = os.path.join(root_directory, 'sentinel-1')
    if not os.path.isdir(s2_base_path) or not os.path.isdir(s1_base_path): return {}
    s2_clusters = [d.path for d in os.scandir(s2_base_path) if d.is_dir()]
    if not s2_clusters: return {}
    all_results = {}
    for s2_cluster_path in s2_clusters:
        cluster_name = os.path.basename(s2_cluster_path)
        s1_cluster_path = os.path.join(s1_base_path, cluster_name)
        if not os.path.isdir(s1_cluster_path): continue
        s1_files = glob.glob(os.path.join(s1_cluster_path, '**', '*.tif'), recursive=True)
        s2_files = glob.glob(os.path.join(s2_cluster_path, '**', '*.tif'), recursive=True)
        if not s1_files or not s2_files: continue
        s1_timed_files = [(p, get_date_from_filename(p)) for p in s1_files if get_date_from_filename(p)]
        if not s1_timed_files: continue
        cluster_pairs = []
        for s2_path in s2_files:
            # to be removed
            # if "QQJ" in s2_path:
            s2_date = get_date_from_filename(s2_path)
            if not s2_date: continue
            nearest_s1_path = min(s1_timed_files, key=lambda x: abs(s2_date - x[1]))[0]
            cluster_pairs.append((nearest_s1_path, s2_path))
        all_results[cluster_name] = cluster_pairs
    return all_results

# class InferenceDataset(Dataset):
#     def __init__(self, file_pairs, config, cloud_detector):
#         self.file_pairs = file_pairs
#         self.config = config
#         self.cloud_detector = cloud_detector

#     def __len__(self):
#         return len(self.file_pairs)

#     def __getitem__(self, idx):
#         s1_path, s2_cloudy_path = self.file_pairs[idx]

#         s2_cloudy_tif_obj, s2_cloudy_img_raw = read_tif(s2_cloudy_path)
#         _, s1_img_raw = read_tif(s1_path)

#         mask_array = get_cloud_map(s2_cloudy_img_raw, self.cloud_detector)
#         s1_processed = process_SAR(s1_img_raw)
#         s2_cloudy_processed = process_MS(s2_cloudy_img_raw)

#         if self.config.use_sar:
#             input_data = np.concatenate((s1_processed, s2_cloudy_processed), axis=0)
#         else:
#             input_data = s2_cloudy_processed
        
#         return torch.from_numpy(input_data), torch.from_numpy(mask_array)

# class InferenceDataset(Dataset):
#     def __init__(self, file_pairs, config, cloud_detector):
#         self.file_pairs = file_pairs
#         self.config = config
#         self.cloud_detector = cloud_detector
#         self.target_size = 256 

#     def __len__(self):
#         return len(self.file_pairs)

#     def __getitem__(self, idx):
#         s1_path, s2_cloudy_path = self.file_pairs[idx]

#         s2_cloudy_tif_obj, s2_cloudy_img_raw = read_tif(s2_cloudy_path)
#         _, s1_img_raw = read_tif(s1_path)
        
#         _, height, width = s2_cloudy_img_raw.shape
        
#         if height > self.target_size or width > self.target_size:

#             top = np.random.randint(0, height - self.target_size + 1)
#             left = np.random.randint(0, width - self.target_size + 1)
            
#             s2_cloudy_img_raw = s2_cloudy_img_raw[:, top:top + self.target_size, left:left + self.target_size]
#             s1_img_raw = s1_img_raw[:, top:top + self.target_size, left:left + self.target_size]

#         mask_array = get_cloud_map(s2_cloudy_img_raw, self.cloud_detector)
#         s1_processed = process_SAR(s1_img_raw)
#         s2_cloudy_processed = process_MS(s2_cloudy_img_raw)
#         print("Shapes:", s1_processed.shape, s2_cloudy_processed.shape, mask_array.shape)
#         if self.config.use_sar:
#             input_data = np.concatenate((s1_processed, s2_cloudy_processed), axis=0)
#         else:
#             input_data = s2_cloudy_processed
        
#         return torch.from_numpy(input_data), torch.from_numpy(mask_array)

def collate_fn(batch):
    """
    Filters out None values from a batch.
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None)

class InferenceDataset(Dataset):
    def __init__(self, file_pairs, config, cloud_detector):
        self.file_pairs = file_pairs
        self.config = config
        self.cloud_detector = cloud_detector
        self.target_size = 256

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        try:
            s1_path, s2_cloudy_path = self.file_pairs[idx]

            s2_cloudy_tif_obj, s2_cloudy_img_raw = read_tif(s2_cloudy_path)
            _, s1_img_raw = read_tif(s1_path)
            

            _, s2_height, s2_width = s2_cloudy_img_raw.shape
            _, s1_height, s1_width = s1_img_raw.shape

            if s2_height < self.target_size or s2_width < self.target_size or \
               s1_height < self.target_size or s1_width < self.target_size:
                return None 

            center_h, center_w = s2_height // 2, s2_width // 2
            half_size = self.target_size // 2
            
            top = center_h - half_size
            left = center_w - half_size
            
            s2_cropped = s2_cloudy_img_raw[:, top:top + self.target_size, left:left + self.target_size]
            s1_cropped = s1_img_raw[:, top:top + self.target_size, left:left + self.target_size]
            

            mask_array = get_cloud_map(s2_cropped, self.cloud_detector)
            s1_processed = process_SAR(s1_cropped)
            s2_cloudy_processed = process_MS(s2_cropped)

            if self.config.use_sar:
                input_data = np.concatenate((s1_processed, s2_cloudy_processed), axis=0)
            else:
                input_data = s2_cloudy_processed
            
            return torch.from_numpy(input_data), torch.from_numpy(mask_array)
        
        except Exception as e:
            return None

conf_path = os.path.join(dirname, test_config.weight_folder, test_config.experiment_name, "conf.json") if not test_config.load_config else test_config.load_config
if not os.path.exists(conf_path):
    raise FileNotFoundError(f"Configuration file not found at {conf_path}")

with open(conf_path) as file:
    model_config = json.loads(file.read())
    t_args = argparse.Namespace()

    no_overwrite = ['pid', 'device', 'resume_at', 'trained_checkp', 'res_dir', 'weight_folder', 'root1', 'root2', 'root3', 
    'max_samples_count', 'batch_size', 'display_step', 'plot_every', 'export_every', 'input_t', 'region', 'min_cov', 'max_cov']
    conf_dict = {key:val for key,val in model_config.items() if key not in no_overwrite}
    for key, val in vars(test_config).items(): 
        if key in no_overwrite: conf_dict[key] = val

    t_args.__dict__.update(conf_dict)
    config = parser.parse_args(namespace=t_args)
    config = utils.str2list(t_args, ["encoder_widths", "decoder_widths", "out_conv"])

if config.pretrain: config.batch_size = 32

def run_batch_inference(config, all_pairs_by_cluster):
    device = torch.device(config.device)

    model = get_model(config)
    model = model.to(device)

    config.N_params = utils.get_ntrainparams(model)
    print(f"TOTAL TRAINABLE PARAMETERS: {config.N_params}\n")
    print(model)
    # load_checkpoint(config, config.model_dir, model, "model")
    ckpt_n = f'_epoch_{config.resume_at}' if config.resume_at > 0 else ''
    load_checkpoint(config, config.weight_folder, model, f"model{ckpt_n}")

    model.eval()
    print("--- Model loaded successfully ---")

    cloud_detector = S2PixelCloudDetector(threshold=0.4, all_bands=True, average_over=4, dilation_size=2)
    all_pairs_by_cluster = all_pairs_by_cluster
    print("----------", all_pairs_by_cluster.keys())

    for cluster_name, pairs_in_cluster in all_pairs_by_cluster.items():
        print(f"\n--- Processing cluster: {cluster_name} ---")
        # to be removed
        # if cluster_name == "tile_1":
        #     continue

        if not pairs_in_cluster:
            print("No pairs to process in this cluster.")
            continue
            
        dataset = InferenceDataset(pairs_in_cluster, config, cloud_detector)
        data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

        for i, (input_batch, mask_batch) in enumerate(tqdm(data_loader, desc=f"Reconstructing {cluster_name}")):
            if input_batch is None:
                continue
            input_tensor = input_batch.unsqueeze(1).to(device)
            mask_tensor = mask_batch.unsqueeze(1).to(device)
            
            with torch.no_grad():
                inputs = {'A': input_tensor, 'B': None, 'dates': None, 'masks': mask_tensor}
                model.set_input(inputs)
                model.forward()
                model.rescale()
                reconstructed_batch = model.fake_B

            if reconstructed_batch.shape[2] > 13:
                image_batch = reconstructed_batch[:, :, :13, :, :]
            else:
                image_batch = reconstructed_batch
            
            output_batch = image_batch.cpu().squeeze(1).numpy()
            
            current_batch_size = output_batch.shape[0]
            for j in range(current_batch_size):
                output_array = np.clip(output_batch[j], 0, 1) * 10000.0
                
                original_index = i * config.batch_size + j
                s1_path, s2_path = pairs_in_cluster[original_index]
                
                with rasterio.open(s2_path) as src:
                    item_meta = src.meta.copy()

                cluster_output_dir = os.path.join(config.res_dir, "clusters", cluster_name)
                os.makedirs(cluster_output_dir, exist_ok=True)
                output_filename = os.path.basename(s2_path)
                output_path = os.path.join(cluster_output_dir, output_filename)
                
                save_geotiff(output_path, output_array, item_meta)

    print(f"\n--- Batch inference complete. Results saved to: {config.res_dir} ---")

def post_process_and_clip_fields(config, all_pairs_by_cluster):
    """
    Finds reconstructed tiles, filters them by field-specific date ranges,
    and clips the field geometry using the rasterize and multiply method.
    """

    all_fields_by_cluster = create_cluster_field_mapping(config.root3, config.field_data_file)
    print("\n--- Starting field clipping process ---")
    print([(cluster, len(fields)) for cluster, fields in all_fields_by_cluster.items()])
    if not all_fields_by_cluster:
        print("Could not load field data. Aborting clipping.")
        return

    fields_base_dir = os.path.join(config.res_dir, "fields")
    os.makedirs(fields_base_dir, exist_ok=True)

    s1_base_path = os.path.join(config.root3, 'sentinel-1')
    all_files = os.listdir(s1_base_path)
    if len(all_files)>1:
        prefix = os.path.commonprefix(all_files)
    elif len(all_files)==1:
        prefix = re.sub(r'\d', '', all_files[0])
    else:
        prefix = ""

    print(f"Using prefix '{prefix}' for cluster directories.")
    
    for cluster_name, fields_in_cluster in tqdm(all_fields_by_cluster.items(), desc="Processing Clusters"):
        reconstructed_tile_dir = os.path.join(config.res_dir, "clusters", prefix+str(cluster_name))
        s2_to_s1_map = {os.path.basename(s2): s1 for s1, s2 in all_pairs_by_cluster.get(prefix+str(cluster_name), [])}
        # should be removed later
        # if cluster_name!=0:
        #     continue
        if not os.path.isdir(reconstructed_tile_dir):
            print(f"Warning: No reconstructed tiles found for cluster {cluster_name}. Skipping.")
            continue
        print(f"--- Processing cluster: {cluster_name} ---")
        reconstructed_tiles = glob.glob(os.path.join(reconstructed_tile_dir, '*.tif'))
        
        for field_data in fields_in_cluster:
            field_id = field_data.get('field_id')
            
            field_output_dir = os.path.join(fields_base_dir, prefix+str(cluster_name), str(field_id))
            os.makedirs(field_output_dir, exist_ok=True)
            os.makedirs(os.path.join(field_output_dir, "s2"), exist_ok=True)
            os.makedirs(os.path.join(field_output_dir, "s1"), exist_ok=True)
            print(f"Created output directory for field {field_id} of cluster {cluster_name}")

            try:
                start_date = datetime.strptime(field_data['image_start_date'], '%Y-%m-%d')
                end_date = datetime.strptime(field_data['image_end_date'], '%Y-%m-%d')
            except (ValueError, TypeError):
                print(f"Warning: Invalid date format for field {field_id}. Skipping.")
                continue

            for tile_path in reconstructed_tiles:
                tile_date = get_date_from_filename(tile_path)
                if tile_date and start_date <= tile_date <= end_date:
                    s1_tile_path = s2_to_s1_map.get(os.path.basename(tile_path))
                    if not s1_tile_path:
                        print(f"Warning: No matching S1 tile found for {os.path.basename(tile_path)}. Skipping.")
                        continue
                    try:
                        output_filename = f"{os.path.basename(tile_path)}"
                        with rasterio.open(tile_path) as src:
                            tile_image_data = src.read()
                            tile_metadata = src.meta.copy()

                            geojson_crs = "EPSG:4326"
                            transformer = Transformer.from_crs(geojson_crs, src.crs, always_xy=True)
                            field_polygon_latlon = shape(field_data['geo_json'])
                            projected_polygon = shapely_transform(transformer.transform, field_polygon_latlon)

                            clipped_field_image, out_transform = mask(src, [projected_polygon], crop=True, nodata=0)
                            
                            tile_metadata = src.meta.copy()
                            tile_metadata.update({
                                "driver": "GTiff",
                                "height": clipped_field_image.shape[1],
                                "width": clipped_field_image.shape[2],
                                "transform": out_transform,
                                "nodata": 0
                            })
                            
                            # mask_array_2d = rasterize(
                            #     shapes=[projected_polygon],
                            #     out_shape=(src.height, src.width),
                            #     transform=src.transform,
                            #     fill=0,
                            #     all_touched=True,
                            #     dtype=rasterio.uint8
                            # )
                            
                            # mask_array_3d = mask_array_2d[np.newaxis, :, :]
                            # clipped_field_image = tile_image_data * mask_array_3d


                            
                            output_path = os.path.join(field_output_dir, "s2", output_filename)
                            
                            save_geotiff(output_path, clipped_field_image, tile_metadata)
                        
                        with rasterio.open(s1_tile_path) as src:

                            # s1_tile_image_data = src.read()
                            s1_tile_metadata = src.meta.copy()

                            geojson_crs = "EPSG:4326"
                            transformer = Transformer.from_crs(geojson_crs, src.crs, always_xy=True)
                            field_polygon_latlon = shape(field_data['geo_json'])
                            projected_polygon = shapely_transform(transformer.transform, field_polygon_latlon)

                            s1_clipped_field_image, s1_out_transform = mask(src, [projected_polygon], crop=True, nodata=0)
                            
                            s1_tile_metadata.update({
                                "driver": "GTiff",
                                "height": s1_clipped_field_image.shape[1],
                                "width": s1_clipped_field_image.shape[2],
                                "transform": s1_out_transform,
                                "nodata": 0
                            })
                            
                            # mask_array_2d = rasterize(
                            #     shapes=[projected_polygon],
                            #     out_shape=(src.height, src.width),
                            #     transform=src.transform,
                            #     fill=0,
                            #     all_touched=True,
                            #     dtype=rasterio.uint8
                            # )
                            
                            # mask_array_3d = mask_array_2d[np.newaxis, :, :]
                            # clipped_field_image = tile_image_data * mask_array_3d

                            s1_output_path = os.path.join(field_output_dir, "s1", output_filename)
                            
                            save_geotiff(s1_output_path, s1_clipped_field_image, s1_tile_metadata)
                            
                    except Exception as e:
                        print(f"Failed to clip field {field_id} from {os.path.basename(tile_path)}. Error: {e}")

    print("\n--- Field clipping complete. ---")


def spatial_sample_stats(ndvi_array, valid_mask):
    """
    Safely takes 5 distinct spatial samples using an intelligent, tiered 
    window size to handle a wide range of small field dimensions.
    """
    spatial_stats = {}
    valid_coords = np.argwhere(valid_mask)
    if valid_coords.shape[0] == 0:
        for i in range(5):
            spatial_stats[f'max_sample_{i+1}'] = np.nan
        return spatial_stats

    min_row, min_col = valid_coords.min(axis=0)
    max_row, max_col = valid_coords.max(axis=0)
    valid_height = max_row - min_row + 1
    valid_width = max_col - min_col + 1
    min_dim = min(valid_height, valid_width)
    # alternative, could try is dim/2+1
    if min_dim > 10:
        window_dim = max(3, int(min_dim / 5))
    elif min_dim > 4:
        window_dim = 3
    else:
        window_dim = 1
        
    if window_dim > 1 and window_dim % 2 == 0:
        window_dim -= 1
    half_win = window_dim // 2

    center_row, center_col = np.mean(valid_coords, axis=0)
    target_points = np.array([
        [min_row, min_col], [min_row, max_col], [center_row, center_col],
        [max_row, min_col], [max_row, max_col]
    ])
    distances = cdist(target_points, valid_coords)
    closest_indices = np.argmin(distances, axis=1)
    sample_coords = valid_coords[closest_indices]

    sample_names = ['sample_1_TL', 'sample_2_TR', 'sample_3_C', 'sample_4_BL', 'sample_5_BR']

    for i, (r, c) in enumerate(sample_coords):
        y_start = max(0, r - half_win)
        y_end = min(ndvi_array.shape[0], r + half_win + 1)
        x_start = max(0, c - half_win)
        x_end = min(ndvi_array.shape[1], c + half_win + 1)
        
        ndvi_window = ndvi_array[y_start:y_end, x_start:x_end]
        mask_window = valid_mask[y_start:y_end, x_start:x_end]
        valid_pixels_in_window = ndvi_window[mask_window]
        
        if valid_pixels_in_window.size > 0:
            spatial_stats[f'max_{sample_names[i]}'] = np.max(valid_pixels_in_window)
        else:
            spatial_stats[f'max_{sample_names[i]}'] = ndvi_array[r, c]
            
    return spatial_stats


def calculate_ndvi_stats(tiff_path):
    """
    Opens a GeoTIFF, calculates NDVI for Sentinel-2, and computes statistics.
    Sentinel-2 bands: Red=B4, NIR=B8.
    """
    try:
        with rasterio.open(tiff_path) as src:
            # Read Red (Band 4) and NIR (Band 8)
            red = src.read(4).astype(np.float32)
            nir = src.read(8).astype(np.float32)

            mask = red != 0
            
            np.seterr(divide='ignore', invalid='ignore')
            ndvi = np.where(
                (nir + red) > 0,
                (nir - red) / (nir + red),
                0  # Set NDVI to 0 where the denominator is zero
            )
            
            valid_ndvi = ndvi[mask]
            
            if valid_ndvi.size == 0:
                return None 

            stats = {
                'mean': np.mean(valid_ndvi),
                'median': np.median(valid_ndvi),
                '25th': np.percentile(valid_ndvi, 25),
                '50th': np.percentile(valid_ndvi, 50),
                '75th': np.percentile(valid_ndvi, 75),
                '90th': np.percentile(valid_ndvi, 90)
            }
            return stats
            
    except Exception as e:
        print(f"Warning: Could not process {tiff_path}. Error: {e}")
        return None


def get_ordinal_suffix(n):
    """Returns the ordinal suffix for a number (e.g., 1st, 2nd, 3rd, 4th)."""
    if 11 <= (n % 100) <= 13:
        return f'{n}th'
    else:
        return f"{n}{'st' if n % 10 == 1 else 'nd' if n % 10 == 2 else 'rd' if n % 10 == 3 else 'th'}"


# def calculate_s2_stats(tiff_path):
#     """Calculates NDVI statistics from a Sentinel-2 TIFF file."""
#     try:
#         with rasterio.open(tiff_path) as src:
#             # Assuming standard band order: Red=B4, NIR=B8
#             red = src.read(4).astype(np.float32)
#             nir = src.read(8).astype(np.float32)
            
#             denominator = nir + red
#             ndvi = np.divide(nir - red, denominator, out=np.full_like(denominator, np.nan), where=denominator!=0)
            
#             if src.nodata is not None:
#                 ndvi[ndvi == src.nodata] = np.nan
            
#             return {
#                 'mean': np.nanmean(ndvi), 'median': np.nanmedian(ndvi),
#                 '25th': np.nanpercentile(ndvi, 25), '50th': np.nanpercentile(ndvi, 50),
#                 '75th': np.nanpercentile(ndvi, 75), '90th': np.nanpercentile(ndvi, 90)
#             }
#     except Exception:
#         return None

def calculate_s2_stats(tiff_path):
    """
    Calculates the MEAN of a comprehensive set of Sentinel-2 vegetation indices.
    NOTE: Assumes S2 band order: B2=Blue, B3=Green, B4=Red, B5=Red Edge 1, 
                                 B8=NIR, B11=SWIR 1.
    """
    try:
        with rasterio.open(tiff_path) as src:
            # --- 1. Read all required bands ---
            blue = src.read(2).astype(np.float32)
            green = src.read(3).astype(np.float32)
            red = src.read(4).astype(np.float32)
            red_edge1 = src.read(5).astype(np.float32)
            nir = src.read(8).astype(np.float32)
            swir1 = src.read(11).astype(np.float32)

            s2_indices = {}
            
            # Greenness Indices
            s2_indices['s2_ndvi'] = np.divide(nir - red, nir + red, out=np.full_like(red, np.nan), where=(nir + red)!=0)
            s2_indices['s2_evi'] = 2.5 * np.divide(nir - red, nir + 6 * red - 7.5 * blue + 1, out=np.full_like(red, np.nan), where=(nir + 6 * red - 7.5 * blue + 1)!=0)
            s2_indices['s2_savi'] = 1.5 * np.divide(nir - red, nir + red + 0.5, out=np.full_like(red, np.nan), where=(nir + red + 0.5)!=0)
            s2_indices['s2_gndvi'] = np.divide(nir - green, nir + green, out=np.full_like(red, np.nan), where=(nir + green)!=0)
            
            # Leaf Pigment Indices
            s2_indices['s2_ci_green'] = np.divide(nir, green, out=np.full_like(red, np.nan), where=green!=0) - 1
            s2_indices['s2_ci_rededge'] = np.divide(nir, red_edge1, out=np.full_like(red, np.nan), where=red_edge1!=0) - 1
            s2_indices['s2_ndre'] = np.divide(nir - red_edge1, nir + red_edge1, out=np.full_like(red, np.nan), where=(nir + red_edge1)!=0)

            # Water and Moisture Stress Indices
            s2_indices['s2_ndwi'] = np.divide(green - nir, green + nir, out=np.full_like(red, np.nan), where=(green + nir)!=0)
            s2_indices['s2_ndmi'] = np.divide(nir - swir1, nir + swir1, out=np.full_like(red, np.nan), where=(nir + swir1)!=0)
            s2_indices['s2_msi'] = np.divide(swir1, nir, out=np.full_like(red, np.nan), where=nir!=0)
            s2_indices['s2_mndwi'] = np.divide(green - swir1, green + swir1, out=np.full_like(red, np.nan), where=(green + swir1)!=0)
            
            
            valid_mask = red != 0
            np.seterr(divide='ignore', invalid='ignore')
            ndvi = s2_indices['s2_ndvi']
            spatial_stats = spatial_sample_stats(ndvi, valid_mask)
            s2_indices.update(spatial_stats)
            mean_stats = {key: np.nanmean(value) for key, value in s2_indices.items()}
            return mean_stats

    except Exception as e:
        print(f"Warning: Could not process S2 file {os.path.basename(tiff_path)}. Error: {e}")
        return None
    

def calculate_s1_stats(tiff_path):
    """
    Calculates the MEAN of a comprehensive set of Sentinel-1 features from GRD data.
    """
    try:
        with rasterio.open(tiff_path) as src:

            vv_db = src.read(1).astype(np.float32)
            vh_db = src.read(2).astype(np.float32)
            vv_linear = 10**(vv_db / 10.0)
            vh_linear = 10**(vh_db / 10.0)

            # Basic Ratios
            denominator_rvi = vv_linear + vh_linear
            rvi = np.divide(4 * vh_linear, denominator_rvi, out=np.full_like(denominator_rvi, np.nan), where=denominator_rvi!=0)
            cross_ratio = np.divide(vh_linear, vv_linear, out=np.full_like(vv_linear, np.nan), where=vv_linear!=0)
            
            # RFDI (Radar Forest Degradation Index)
            rfdi_num = vv_linear - vh_linear
            rfdi_den = vv_linear + vh_linear
            rfdi = np.divide(rfdi_num, rfdi_den, out=np.full_like(rfdi_den, np.nan), where=rfdi_den!=0)

            # RVI4S1 (Radar Vegetation Index for Sentinel-1)
            # Based on the document, q is the ratio of cross-pol to co-pol (VH/VV) 
            q = cross_ratio 
            rvi4s1_num = q * (q + 3)
            rvi4s1_den = (q + 1)**2
            rvi4s1 = np.divide(rvi4s1_num, rvi4s1_den, out=np.full_like(rvi4s1_den, np.nan), where=rvi4s1_den!=0)

            s1_features = {
                's1_vv_linear': vv_linear,
                's1_vh_linear': vh_linear,
                's1_rvi': rvi,
                's1_cross_ratio': cross_ratio,
                's1_rfdi': rfdi,
                's1_rvi4s1': rvi4s1,
            }
            
            for band_db, name in [(vv_db, 'vv'), (vh_db, 'vh')]:
                band_db[np.isnan(band_db)] = 0
                img_8bit = np.uint8(255 * (band_db - np.min(band_db)) / np.ptp(band_db))
                
                glcm = graycomatrix(img_8bit, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
                
                s1_features[f's1_{name}_homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
                s1_features[f's1_{name}_contrast'] = graycoprops(glcm, 'contrast')[0, 0]

            mean_stats = {key: np.nanmean(value) for key, value in s1_features.items()}
            return mean_stats
            
    except Exception as e:
        print(f"Warning: Could not process S1 file {os.path.basename(tiff_path)}. Error: {e}")
        return None


def generate_timeseries_report(config):
    """
    Generates a CSV report with NDVI statistics, completely skipping any 
    fields that do not contain any .tif files.
    """
    fields_base_dir = os.path.join(config.res_dir, 'fields')
    output_csv_path = os.path.join(config.res_dir, 'timeseries_report.csv')

    print(f"\n--- Starting Timeseries report generation from: {fields_base_dir} ---")
    
    if not os.path.isdir(fields_base_dir):
        print(f"Error: Fields directory not found at '{fields_base_dir}'. Aborting report generation.")
        return

    field_paths = glob.glob(os.path.join(fields_base_dir, '*', '*'))
    print("fields found:", field_paths[:5])
    all_field_data = []

    for field_path in tqdm(field_paths, desc="Generating Report"):
        if not os.path.isdir(field_path):
            continue
            
        parts = field_path.split(os.sep)
        field_id = parts[-1]
        cluster_id = parts[-2]
        
        tiff_files = sorted(glob.glob(os.path.join(field_path, "s2", '*.tif')), key=get_date_from_filename)
        print(f"DEBUG: Found {len(tiff_files)} TIFF files for field {field_id} in cluster {cluster_id}")
        
        if not tiff_files:
            continue

        field_row = {'cluster_id': cluster_id, 'field_id': field_id}

        for i, s2_tiff_path in enumerate(tiff_files):
            timepoint_str = get_ordinal_suffix(i + 1) + "_timepoint"
            
            # Calculate S2 stats (mean of all indices)
            s2_stats = calculate_s2_stats(s2_tiff_path)
            if s2_stats:
                for key, val in s2_stats.items():
                    field_row[f'{key}_mean_{timepoint_str}'] = val
            
            # Find and Process Corresponding S1 Image
            s2_filename = os.path.basename(s2_tiff_path)
            s1_tiff_path = os.path.join(field_path, "s1", s2_filename)
            
            if os.path.exists(s1_tiff_path):
                # Calculate S1 stats (mean only)
                s1_stats = calculate_s1_stats(s1_tiff_path)
                if s1_stats:
                    for key, val in s1_stats.items():
                        field_row[f'{key}_mean_{timepoint_str}'] = val
            
        all_field_data.append(field_row)

    if not all_field_data:
        print("No field data was processed. The output CSV will be empty.")
        return

    df = pd.DataFrame(all_field_data)
    cols = ['cluster_id', 'field_id'] + sorted([col for col in df.columns if col not in ['cluster_id', 'field_id']])
    df = df[cols]
    
    df.to_csv(output_csv_path, index=False)
    
    print(f"\n--- Full Time-series report complete. Saved to: {output_csv_path} ---")
    print(f"Processed and included {len(df)} fields with valid images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch Inference Script')
    parser.add_argument('--weight_folder', type=str, required=True, help='Path to the trained model experiment directory')
    parser.add_argument('--root3', type=str, required=True, help='Path to the root folder of new data.')
    parser.add_argument('--res_dir', type=str, required=True, help='Path to the root folder where results will be saved.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device for inference')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of samples to process at once')
    parser.add_argument('--field_data_file', type=str, required=True, help='Filepath of the CSV or Excel file containing field data')
    
    all_pairs_by_cluster = find_input_pairs(config.root3)
    run_batch_inference(config, all_pairs_by_cluster)
    post_process_and_clip_fields(config, all_pairs_by_cluster)
    generate_timeseries_report(config)