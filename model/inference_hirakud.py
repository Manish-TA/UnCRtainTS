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

def save_reconstructed_tif(path, array, source_tif_meta):
    """Saves a numpy array as a georeferenced TIFF file."""
    metadata = source_tif_meta.copy()
    # Update metadata for the 13-band S2 output
    metadata.update({
        'dtype': 'float32',
        'count': array.shape[0]
    })
    with rasterio.open(path, 'w', **metadata) as dst:
        dst.write(array)
    
def save_geotiff(path, array, metadata):
    metadata.update({'dtype': array.dtype.name, 'count': array.shape[0]})
    with rasterio.open(path, 'w', **metadata) as dst:
        dst.write(array)


def get_date_from_filename(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'_(\d{8}T\d{6})_', filename)
    if match: return datetime.strptime(match.group(1), '%Y%m%dT%H%M%S')
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
        
        if cluster_id:
            try:
                field_info = {
                    'field_id': row.get('field_id'),
                    'geo_json': json.loads(row.get('geo_json')),
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
            s2_date = get_date_from_filename(s2_path)
            if not s2_date: continue
            nearest_s1_path = min(s1_timed_files, key=lambda x: abs(s2_date - x[1]))[0]
            cluster_pairs.append((nearest_s1_path, s2_path))
        all_results[cluster_name] = cluster_pairs
    return all_results

class InferenceDataset(Dataset):
    def __init__(self, file_pairs, config, cloud_detector):
        self.file_pairs = file_pairs
        self.config = config
        self.cloud_detector = cloud_detector

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        s1_path, s2_cloudy_path = self.file_pairs[idx]

        s2_cloudy_tif_obj, s2_cloudy_img_raw = read_tif(s2_cloudy_path)
        _, s1_img_raw = read_tif(s1_path)

        mask_array = get_cloud_map(s2_cloudy_img_raw, self.cloud_detector)
        s1_processed = process_SAR(s1_img_raw)
        s2_cloudy_processed = process_MS(s2_cloudy_img_raw)

        if self.config.use_sar:
            input_data = np.concatenate((s1_processed, s2_cloudy_processed), axis=0)
        else:
            input_data = s2_cloudy_processed
        
        return torch.from_numpy(input_data), torch.from_numpy(mask_array)

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

def run_batch_inference(config):
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
    all_pairs_by_cluster = find_input_pairs(config.root3)

    for cluster_name, pairs_in_cluster in all_pairs_by_cluster.items():
        print(f"\n--- Processing cluster: {cluster_name} ---")
        if not pairs_in_cluster:
            print("No pairs to process in this cluster.")
            continue
            
        dataset = InferenceDataset(pairs_in_cluster, config, cloud_detector)
        data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

        for i, (input_batch, mask_batch) in enumerate(tqdm(data_loader, desc=f"Reconstructing {cluster_name}")):
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

                cluster_output_dir = os.path.join(config.res_dir, cluster_name)
                os.makedirs(cluster_output_dir, exist_ok=True)
                output_filename = os.path.basename(s2_path)
                output_path = os.path.join(cluster_output_dir, output_filename)
                
                save_reconstructed_tif(output_path, output_array, item_meta)

    print(f"\n--- Batch inference complete. Results saved to: {config.res_dir} ---")

def post_process_and_clip_fields(config):
    """
    Finds reconstructed tiles, filters them by field-specific date ranges,
    and clips the field geometry using the rasterize and multiply method.
    """

    all_fields_by_cluster = create_cluster_field_mapping(config.input_dir, config.field_data_file)
    if not all_fields_by_cluster:
        print("Could not load field data. Aborting clipping.")
        return

    fields_base_dir = os.path.join(config.output_dir, "fields")
    os.makedirs(fields_base_dir, exist_ok=True)
    
    for cluster_name, fields_in_cluster in tqdm(all_fields_by_cluster.items(), desc="Processing Clusters"):
        reconstructed_tile_dir = os.path.join(config.output_dir, cluster_name)
        if not os.path.isdir(reconstructed_tile_dir):
            print(f"Warning: No reconstructed tiles found for cluster {cluster_name}. Skipping.")
            continue
            
        reconstructed_tiles = glob.glob(os.path.join(reconstructed_tile_dir, '*.tif'))
        
        for field_data in fields_in_cluster:
            field_id = field_data.get('field_id')
            
            field_output_dir = os.path.join(fields_base_dir, cluster_name, str(field_id))
            os.makedirs(field_output_dir, exist_ok=True)
            
            try:
                start_date = datetime.strptime(field_data['image_start_date'], '%Y-%m-%d')
                end_date = datetime.strptime(field_data['image_end_date'], '%Y-%m-%d')
            except (ValueError, TypeError):
                print(f"Warning: Invalid date format for field {field_id}. Skipping.")
                continue

            for tile_path in reconstructed_tiles:
                tile_date = get_date_from_filename(tile_path)
                if tile_date and start_date <= tile_date <= end_date:
                    try:
                        with rasterio.open(tile_path) as src:
                            tile_image_data = src.read()
                            tile_metadata = src.meta.copy()

                            geojson_crs = "EPSG:4326"
                            transformer = Transformer.from_crs(geojson_crs, src.crs, always_xy=True)
                            field_polygon_latlon = shape(json.loads(field_data['geo_json']))
                            projected_polygon = shapely_transform(transformer.transform, field_polygon_latlon)
                            
                            mask_array_2d = rasterize(
                                shapes=[projected_polygon],
                                out_shape=(src.height, src.width),
                                transform=src.transform,
                                fill=0,
                                all_touched=True,
                                dtype=rasterio.uint8
                            )
                            
                            mask_array_3d = mask_array_2d[np.newaxis, :, :]
                            
                            clipped_field_image = tile_image_data * mask_array_3d

                            output_filename = f"{field_id}_{os.path.basename(tile_path)}"
                            output_path = os.path.join(field_output_dir, output_filename)
                            
                            save_geotiff(output_path, clipped_field_image, tile_metadata)
                            
                    except Exception as e:
                        print(f"Failed to clip field {field_id} from {os.path.basename(tile_path)}. Error: {e}")

    print("\n--- Field clipping complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch Inference Script')
    parser.add_argument('--weight_folder', type=str, required=True, help='Path to the trained model experiment directory')
    parser.add_argument('--root3', type=str, required=True, help='Path to the root folder of new data.')
    parser.add_argument('--res_dir', type=str, required=True, help='Path to the root folder where results will be saved.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device for inference')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of samples to process at once')
    parser.add_argument('--field_data_file', type=str, required=True, help='Filepath of the CSV or Excel file containing field data')
    
    run_batch_inference(config)
    post_process_and_clip_fields(config)