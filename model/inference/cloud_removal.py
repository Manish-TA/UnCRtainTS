import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import glob
import re
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict 
from s2cloudless import S2PixelCloudDetector
import numpy as np
from src.model_utils import get_model, load_checkpoint
from src import utils

dirname = os.path.dirname(os.path.abspath(__file__))


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


def run_batch_inference(config, all_pairs_by_cluster):
    device = torch.device(config.device)

    model = get_model(config)
    model = model.to(device)

    config.N_params = utils.get_ntrainparams(model)
    print(f"TOTAL TRAINABLE PARAMETERS: {config.N_params}\n")
    print(model)
    # load_checkpoint(config, config.model_dir, model, "model")
    ckpt_n = f'_epoch_{config.resume_at}' if config.resume_at > 0 else ''
    load_checkpoint(config, os.path.join(dirname, config.weight_folder), model, f"model{ckpt_n}")

    model.eval()
    print("--- Model loaded successfully ---")

    cloud_detector = S2PixelCloudDetector(threshold=0.4, all_bands=True, average_over=4, dilation_size=2)
    all_pairs_by_cluster = all_pairs_by_cluster
    print("----------", all_pairs_by_cluster.keys())

    for cluster_name, pairs_in_cluster in all_pairs_by_cluster.items():
        print(f"\n--- Processing cluster: {cluster_name} ---")

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
