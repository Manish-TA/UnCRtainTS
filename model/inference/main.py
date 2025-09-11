import os
import sys

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(dirname))

import json
import argparse
import torch
from src import utils
from parse_args import create_parser
from post_process import post_process_and_clip_fields, generate_timeseries_report
from cloud_removal import find_input_pairs, run_batch_inference

parser = create_parser(mode='test')
test_config = parser.parse_args()
test_config.pid = os.getpid()


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