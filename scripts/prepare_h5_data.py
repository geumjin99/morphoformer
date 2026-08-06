"""Build the HDF5 splits from SHAFTS raster products and Sentinel imagery.

This script converts the raw GeoTIFF raster data from the SHAFTS project
(https://doi.org/10.5281/zenodo.6370003) into the HDF5 layout this repository
expects.

Data sources:
  - Building Height / Footprint: SHAFTS 100m rasters (Zenodo: 10.5281/zenodo.6370003)
  - Sentinel-1: SAR composite (VH, VV) - 50th percentile annual
  - Sentinel-2: Optical composite (B, G, R, NIR) - 50th percentile annual
  - DEM: SRTM 30m (resampled to 100m)

Usage:
  python scripts/prepare_h5_data.py \
      --data-info-json /path/to/data_info.json \
      --satellite-dir /path/to/SatelliteDataDirectory \
      --building-dir /path/to/BuildingInfoDirectory \
      --srtm-dir /path/to/SRTM \
      --output-dir /path/to/output

The data_info.json should contain one JSON object per line with fields:
  {"region": "CityName", "year": 2020, "extent": [lon_min, lat_min, lon_max, lat_max]}
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    print("ERROR: GDAL (python-gdal / osgeo) is required. Install via pip or conda.")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate H5 training data from SHAFTS rasters')

    parser.add_argument('--data-info-json', type=str, required=True,
                       help='Path to data_info.json (per-line JSON with region/year/extent)')
    parser.add_argument('--satellite-dir', type=str, required=True,
                       help='Root directory of satellite data (sentinel_1_50pt/, sentinel_2_50pt/)')
    parser.add_argument('--building-dir', type=str, required=True,
                       help='Root directory of building data (BuildingHeight_100m/, BuildingFootprint_100m/)')
    parser.add_argument('--srtm-dir', type=str, required=True,
                       help='Directory containing SRTM DEM tiles')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for H5 files')
    parser.add_argument('--resolution', type=int, default=100,
                       help='Target spatial resolution in meters (default: 100)')
    parser.add_argument('--patch-size', type=int, default=10,
                       help='Patch size in pixels (default: 10)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')

    return parser.parse_args()


def read_raster_patch(tif_path, row, col, patch_size):
    """Read a patch from a GeoTIFF file."""
    ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
    if ds is None:
        return None

    n_bands = ds.RasterCount
    patch = np.zeros((n_bands, patch_size, patch_size), dtype=np.float32)

    for b in range(n_bands):
        band = ds.GetRasterBand(b + 1)
        data = band.ReadAsArray(col, row, patch_size, patch_size)
        if data is not None:
            patch[b] = data.astype(np.float32)

    ds = None
    return patch


def get_valid_coords(height_path, footprint_path, patch_size=10):
    """Get valid coordinate pairs where both height and footprint data exist."""
    ds_h = gdal.Open(height_path, gdal.GA_ReadOnly)
    ds_f = gdal.Open(footprint_path, gdal.GA_ReadOnly)

    if ds_h is None or ds_f is None:
        return [], None, None

    h_band = ds_h.GetRasterBand(1)
    f_band = ds_f.GetRasterBand(1)
    h_data = h_band.ReadAsArray().astype(np.float32)
    f_data = f_band.ReadAsArray().astype(np.float32)

    n_rows = h_data.shape[0] // patch_size
    n_cols = h_data.shape[1] // patch_size

    coords = []
    bh_values = []
    bf_values = []

    for r in range(n_rows):
        for c in range(n_cols):
            r0, c0 = r * patch_size, c * patch_size
            h_patch = h_data[r0:r0+patch_size, c0:c0+patch_size]
            f_patch = f_data[r0:r0+patch_size, c0:c0+patch_size]

            # Skip patches with nodata or all-zero
            if np.any(np.isnan(h_patch)) or np.all(h_patch == 0):
                continue

            mean_h = np.nanmean(h_patch)
            mean_f = np.nanmean(f_patch)

            coords.append((r, c))
            bh_values.append(mean_h)
            bf_values.append(mean_f)

    ds_h = None
    ds_f = None

    return coords, np.array(bh_values), np.array(bf_values)


def process_city(city_name, building_dir, satellite_dir, srtm_dir, 
                 resolution=100, patch_size=10):
    """Process all satellite bands for a single city."""
    height_path = os.path.join(building_dir, f'BuildingHeight_{resolution}m',
                               f'{city_name}_building_height.tif')
    footprint_path = os.path.join(building_dir, f'BuildingFootprint_{resolution}m',
                                   f'{city_name}_building_footprint.tif')

    if not os.path.exists(height_path) or not os.path.exists(footprint_path):
        print(f"  Skipping {city_name}: missing building rasters")
        return None

    coords, bh, bf = get_valid_coords(height_path, footprint_path, patch_size)
    if len(coords) == 0:
        print(f"  Skipping {city_name}: no valid patches")
        return None

    # Sentinel-1 bands (VH, VV)
    s1_bands = {}
    for b_idx, band_name in enumerate(['B1', 'B2']):
        key = f'sentinel_1_50pt_{band_name}'
        tif = os.path.join(satellite_dir, 'sentinel_1_50pt',
                          f'{city_name}_sentinel_1_50pt.tif')
        if os.path.exists(tif):
            patches = []
            for r, c in coords:
                p = read_raster_patch(tif, r * patch_size, c * patch_size, patch_size)
                if p is not None and p.shape[0] > b_idx:
                    patches.append(p[b_idx:b_idx+1])
                else:
                    patches.append(np.zeros((1, patch_size, patch_size), dtype=np.float32))
            s1_bands[key] = np.stack(patches)

    # Sentinel-2 bands (B, G, R, NIR)
    s2_bands = {}
    for b_idx, band_name in enumerate(['B1', 'B2', 'B3', 'B4']):
        key = f'sentinel_2_50pt_{band_name}'
        tif = os.path.join(satellite_dir, 'sentinel_2_50pt',
                          f'{city_name}_sentinel_2_50pt.tif')
        if os.path.exists(tif):
            patches = []
            for r, c in coords:
                p = read_raster_patch(tif, r * patch_size, c * patch_size, patch_size)
                if p is not None and p.shape[0] > b_idx:
                    patches.append(p[b_idx:b_idx+1])
                else:
                    patches.append(np.zeros((1, patch_size, patch_size), dtype=np.float32))
            s2_bands[key] = np.stack(patches)

    # DEM
    dem_bands = {}
    dem_tif = os.path.join(srtm_dir, f'{city_name}_srtm.tif')
    if os.path.exists(dem_tif):
        patches = []
        for r, c in coords:
            p = read_raster_patch(dem_tif, r * patch_size, c * patch_size, patch_size)
            if p is not None:
                patches.append(p[0:1])
            else:
                patches.append(np.zeros((1, patch_size, patch_size), dtype=np.float32))
        dem_bands['DEM'] = np.stack(patches)

    result = {
        'coords': np.array(coords, dtype=np.int32),
        'BuildingHeight': bh.astype(np.float32),
        'BuildingFootprint': bf.astype(np.float32),
    }
    result.update(s1_bands)
    result.update(s2_bands)
    result.update(dem_bands)

    return result


def save_h5(output_path, city_data_dict):
    """Save all city data into a single H5 file."""
    with h5py.File(output_path, 'w') as f:
        for city_name, data in city_data_dict.items():
            group = f.create_group(city_name)
            for key, arr in data.items():
                group.create_dataset(key, data=arr, compression='gzip',
                                    compression_opts=4)
    print(f"Saved: {output_path} ({len(city_data_dict)} cities)")


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load city list
    cities = []
    with open(args.data_info_json, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    cities.append(item['region'])
                except (json.JSONDecodeError, KeyError):
                    continue

    cities = sorted(set(cities))
    print(f"Processing {len(cities)} cities...")

    # Process all cities
    all_city_data = {}
    for i, city in enumerate(cities):
        print(f"[{i+1}/{len(cities)}] Processing {city}...")
        result = process_city(city, args.building_dir, args.satellite_dir,
                             args.srtm_dir, args.resolution, args.patch_size)
        if result is not None:
            all_city_data[city] = result
            n = len(result['coords'])
            print(f"  {city}: {n} valid patches")

    if not all_city_data:
        print("ERROR: No valid data found.")
        return

    # Split into train/val/test
    city_names = sorted(all_city_data.keys())
    np.random.seed(42)
    np.random.shuffle(city_names)

    n_cities = len(city_names)
    n_train = int(n_cities * args.train_ratio)
    n_val = int(n_cities * args.val_ratio)

    train_cities = city_names[:n_train]
    val_cities = city_names[n_train:n_train + n_val]
    test_cities = city_names[n_train + n_val:]

    print(f"\nSplit: train={len(train_cities)}, val={len(val_cities)}, test={len(test_cities)}")

    # Save H5 files
    save_h5(os.path.join(args.output_dir, 'train.h5'),
            {c: all_city_data[c] for c in train_cities})
    save_h5(os.path.join(args.output_dir, 'valid.h5'),
            {c: all_city_data[c] for c in val_cities})
    save_h5(os.path.join(args.output_dir, 'test.h5'),
            {c: all_city_data[c] for c in test_cities})

    total_patches = sum(len(all_city_data[c]['coords']) for c in city_names)
    print(f"\nDone. Total: {total_patches} patches across {n_cities} cities.")


if __name__ == '__main__':
    main()
