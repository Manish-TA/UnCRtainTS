import os
import numpy as np
import pandas as pd
import rasterio
import glob
import re
from tqdm import tqdm
from datetime import datetime, timedelta
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform
from pyproj import Transformer
from rasterio.mask import mask
from skimage.feature import graycomatrix, graycoprops
import numpy as np
from scipy.spatial.distance import cdist

from cloud_removal import create_cluster_field_mapping, get_date_from_filename, save_geotiff


def get_ordinal_suffix(n):
    """Returns the ordinal suffix for a number (e.g., 1st, 2nd, 3rd, 4th)."""
    if 11 <= (n % 100) <= 13:
        return f'{n}th'
    else:
        return f"{n}{'st' if n % 10 == 1 else 'nd' if n % 10 == 2 else 'rd' if n % 10 == 3 else 'th'}"


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
                            
                            output_path = os.path.join(field_output_dir, "s2", output_filename)
                            
                            save_geotiff(output_path, clipped_field_image, tile_metadata)
                        
                        with rasterio.open(s1_tile_path) as src:
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
                0  
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
                    if "max_sample" in key:
                        field_row[f'{key}_{timepoint_str}'] = val
                    else:
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
    cols = list(all_field_data[0].keys())
    df = df[cols]
    
    df.to_csv(output_csv_path, index=False)
    
    print(f"\n--- Full Time-series report complete. Saved to: {output_csv_path} ---")
    print(f"Processed and included {len(df)} fields with valid images.")
