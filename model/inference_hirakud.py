import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import glob
import re
from tqdm import tqdm
from datetime import datetime, timedelta

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

def get_date_from_filename(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'_(\d{8}T\d{6})_', filename)
    if match: return datetime.strptime(match.group(1), '%Y%m%dT%H%M%S')
    return None

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
        self.target_size = 256 

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        s1_path, s2_cloudy_path = self.file_pairs[idx]

        s2_cloudy_tif_obj, s2_cloudy_img_raw = read_tif(s2_cloudy_path)
        _, s1_img_raw = read_tif(s1_path)
        
        _, height, width = s2_cloudy_img_raw.shape
        
        if height > self.target_size or width > self.target_size:

            top = np.random.randint(0, height - self.target_size + 1)
            left = np.random.randint(0, width - self.target_size + 1)
            
            s2_cloudy_img_raw = s2_cloudy_img_raw[:, top:top + self.target_size, left:left + self.target_size]
            s1_img_raw = s1_img_raw[:, top:top + self.target_size, left:left + self.target_size]

        mask_array = get_cloud_map(s2_cloudy_img_raw, self.cloud_detector)
        s1_processed = process_SAR(s1_img_raw)
        s2_cloudy_processed = process_MS(s2_cloudy_img_raw)

        if self.config.use_sar:
            input_data = np.concatenate((s1_processed, s2_cloudy_processed), axis=0)
        else:
            input_data = s2_cloudy_processed
        
        return {
            'input_tensor': torch.from_numpy(input_data),
            'mask_tensor': torch.from_numpy(mask_array),
            'metadata': s2_cloudy_tif_obj.meta,
            's2_path': s2_cloudy_path
        }

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

    # --- Initialize Cloud Detector and Find Files ---
    cloud_detector = S2PixelCloudDetector(threshold=0.4, all_bands=True, average_over=4, dilation_size=2)
    all_pairs_by_cluster = find_input_pairs(config.root3)

    # --- Main Processing Loop (Outer loop for clusters) ---
    for cluster_name, pairs_in_cluster in all_pairs_by_cluster.items():
        print(f"\n--- Processing cluster: {cluster_name} ---")
        if not pairs_in_cluster:
            print("No pairs to process in this cluster.")
            continue
            
        # --- Create Dataset and DataLoader for the current cluster ---
        dataset = InferenceDataset(pairs_in_cluster, config, cloud_detector)
        data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

        # --- Inner loop for batches ---
        for batch in tqdm(data_loader, desc=f"Reconstructing {cluster_name}"):
            input_tensor = batch['input_tensor'].to(device)
            mask_tensor = batch['mask_tensor'].to(device)
            
            # Add the time dimension the model expects
            input_tensor = input_tensor.unsqueeze(1)
            mask_tensor = mask_tensor.unsqueeze(1)

            with torch.no_grad():
                inputs = {'A': input_tensor, 'B': None, 'dates': None, 'masks': mask_tensor}
                model.set_input(inputs)
                model.forward()
                model.rescale()
                reconstructed_batch = model.fake_B

            # Separate image from variance if necessary
            if reconstructed_batch.shape[2] > 13:
                image_batch = reconstructed_batch[:, :, :13, :, :]
            else:
                image_batch = reconstructed_batch
            
            # Move to CPU for saving
            output_batch = image_batch.cpu().squeeze(1).numpy()
            
            for i in range(output_batch.shape[0]):
                output_array = np.clip(output_batch[i], 0, 1) * 10000.0
                
                item_meta = {}
                for key, value in batch['metadata'].items():
                    item_meta[key] = value[i]

                for key, val in item_meta.items():
                    if torch.is_tensor(val):
                        item_meta[key] = val.item()
                
                s2_path = batch['s2_path'][i]
                cluster_output_dir = os.path.join(config.output_dir, cluster_name)
                os.makedirs(cluster_output_dir, exist_ok=True)
                output_filename = os.path.basename(s2_path)
                output_path = os.path.join(cluster_output_dir, output_filename)
                
                save_reconstructed_tif(output_path, output_array, item_meta)


    print(f"\n--- Batch inference complete. Results saved to: {config.res_dir} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch Inference Script')
    parser.add_argument('--weight_folder', type=str, required=True, help='Path to the trained model experiment directory')
    parser.add_argument('--root3', type=str, required=True, help='Path to the root folder of new data.')
    parser.add_argument('--res_dir', type=str, required=True, help='Path to the root folder where results will be saved.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device for inference')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of samples to process at once')
    
    run_batch_inference(config)