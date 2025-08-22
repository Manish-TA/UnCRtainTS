import os
import sys
import json
import argparse
import numpy as np
import torch
import rasterio
import glob
from tqdm import tqdm
from natsort import natsorted

# --- Make project modules available ---
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(dirname))

from parse_args import create_parser
parser = create_parser(mode='test')
test_config = parser.parse_args()
test_config.pid = os.getpid()


from src.model_utils import get_model, load_checkpoint
from src import utils
from s2cloudless import S2PixelCloudDetector # For on-the-fly cloud masks

def read_tif(path_IMG):
    """Reads a TIFF file, returns the rasterio object and the numpy array."""
    with rasterio.open(path_IMG) as tif:
        return tif, tif.read().astype(np.float32)

def process_MS(img):
    """Applies the default pre-processing to a Sentinel-2 image."""
    intensity_min, intensity_max = 0, 10000
    img = np.clip(img, intensity_min, intensity_max)
    img = (img - intensity_min) / (intensity_max - intensity_min)
    return np.nan_to_num(img)

def process_SAR(img):
    """Applies the default pre-processing to a Sentinel-1 image."""
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
    
    # Add a batch dimension
    cloud_mask = cloud_detector.get_cloud_masks(img_for_detector[None, ...])[0, ...]
    return cloud_mask.astype(np.float32)

def save_reconstructed_tif(path, array, source_tif_meta):
    """Saves a numpy array as a georeferenced TIFF file."""
    metadata = source_tif_meta.copy()
    # Update metadata for the 13-band S2 output
    metadata.update({
        'dtype': 'uint16',
        'count': 13
    })
    with rasterio.open(path, 'w', **metadata) as dst:
        dst.write(array.astype(np.uint16))

def find_input_pairs(input_dir):
    """Finds all corresponding S1 and S2_cloudy pairs in a directory."""
    s1_files = natsorted(glob.glob(os.path.join(input_dir, '**', '*_s1.tif'), recursive=True))
    pairs = []
    for s1_path in s1_files:
        s2_cloudy_path = s1_path.replace('_s1', '_s2_cloudy')
        if os.path.exists(s2_cloudy_path):
            pairs.append((s1_path, s2_cloudy_path))
        else:
            print(f"Warning: Found S1 file but missing corresponding S2_cloudy file: {s1_path}")
    return pairs


conf_path = os.path.join(test_config.model_dir, "conf.json")
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


def run_batch_inference(config):
    device = torch.device(config.device)

    model = get_model(config)
    model = model.to(device)

    config.N_params = utils.get_ntrainparams(model)
    print(f"TOTAL TRAINABLE PARAMETERS: {config.N_params}\n")
    print(model)
    # load_checkpoint(config, config.model_dir, model, "model")
    ckpt_n = f'_epoch_{config.resume_at}' if config.resume_at > 0 else ''
    load_checkpoint(config, config.model_dir, model, f"model{ckpt_n}")

    model.eval()
    print("--- Model loaded successfully ---")

    # --- 2. Initialize Cloud Detector ---
    # This uses the same parameters as your original data loader
    cloud_detector = S2PixelCloudDetector(threshold=0.4, all_bands=True, average_over=4, dilation_size=2)
    print("--- Cloud detector initialized ---")

    # --- 3. Find all data pairs to process ---
    input_pairs = find_input_pairs(config.input_dir)
    if not input_pairs:
        print(f"Error: No S1/S2_cloudy pairs found in {config.input_dir}. Please check the directory structure.")
        return
    print(f"--- Found {len(input_pairs)} pairs to process ---")
    
    # --- 4. Main Processing Loop ---
    for s1_path, s2_cloudy_path in tqdm(input_pairs, desc="Reconstructing Images"):
        try:
            # Load data
            s2_cloudy_tif, s2_cloudy_img_raw = read_tif(s2_cloudy_path)
            _, s1_img_raw = read_tif(s1_path)

            # Generate the cloud mask from the raw S2 cloudy data
            mask_array = get_cloud_map(s2_cloudy_img_raw, cloud_detector)
            
            # Pre-process images for the model
            s1_processed = process_SAR(s1_img_raw)
            s2_cloudy_processed = process_MS(s2_cloudy_img_raw)

            # Combine S1 and S2 if required by the model
            if config.use_sar:
                print("--- Using SAR data also as input ---")
                input_data = np.concatenate((s1_processed, s2_cloudy_processed), axis=0)
            else:
                input_data = s2_cloudy_processed
            
            # Convert to PyTorch tensors and add batch/time dimensions
            input_tensor = torch.from_numpy(input_data).unsqueeze(0).unsqueeze(0).to(device)
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).to(device)

            # Run Inference
            with torch.no_grad():
                inputs = {'A': input_tensor, 'B': None, 'dates': None, 'masks': mask_tensor}
                model.set_input(inputs)
                model.forward()
                reconstructed_tensor = model.fake_B

            # Post-process and save the output
            output_array = reconstructed_tensor.cpu().squeeze().numpy()
            output_array = np.clip(output_array, 0, 1) * 10000.0
            
            # Create a descriptive output filename
            base_name = os.path.basename(s1_path).replace('_s1.tif', '_s2_reconstructed.tif')
            output_path = os.path.join(config.output_dir, base_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the result as a georeferenced TIFF
            save_reconstructed_tif(output_path, output_array, s2_cloudy_tif.meta)

        except Exception as e:
            print(f"\nFailed to process {os.path.basename(s1_path)}. Error: {e}")

    print(f"\n--- Batch inference complete. Results saved to: {config.output_dir} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch Inference Script for SEN12MS-CR Image Reconstruction')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the trained model experiment directory (e.g., ../results/monotemporalL2)')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the root folder containing new, unseen data.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the folder where reconstructed images will be saved.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for inference (cuda or cpu)')
    
    # args = parser.parse_args()
    run_batch_inference(config)