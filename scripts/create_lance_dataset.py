import os
import h5py
import numpy as np
import pyarrow as pa
import lancedb
import time
from tqdm import tqdm
from PIL import Image
import io

import openslide

def find_wsi_file(wsi_dir, wsi_id):
    """
    Find the WSI file corresponding to a given WSI ID.
    """
    return os.path.join(wsi_dir, f"{wsi_id}.tif")


def process_h5_patches(h5_dir, wsi_dir, db_path="./lancedb", table_name="wsi_patches"):
    """
    Process all H5 files in a directory and load patches into LanceDB.
    Reads coordinates from H5 files and extracts patches from corresponding WSI files.
    
    Args:
        h5_dir: Directory containing H5 patch files
        wsi_dir: Directory containing WSI files
        db_path: Path to the LanceDB database directory (default: "./lancedb")
        table_name: Name for the LanceDB table (default: "wsi_patches")
    """
    # Find all H5 files
    h5_files = [f for f in os.listdir(h5_dir) if f.endswith('.h5')]
    
    if not h5_files:
        raise ValueError(f"No H5 files found in {h5_dir}")
    
    print(f"Found {len(h5_files)} H5 files to process")
    
    # Connect to LanceDB
    db = lancedb.connect(db_path)
    
    # Define schema for the table
    schema = pa.schema([
        ('wsi_id', pa.string()),
        ('patch_idx', pa.int64()),
        ('coord_x', pa.int64()),
        ('coord_y', pa.int64()),
        ('image', pa.binary())
    ])
    
    # Create table (will overwrite if exists)
    try:
        table = db.create_table(table_name, schema=schema, mode='overwrite')
        print(f"Created table: {table_name}")
    except Exception as e:
        print(f"Table might already exist, trying to open: {e}")
        table = db.open_table(table_name)
    
    # Process H5 files and insert data
    for h5_filename in tqdm(h5_files, desc="Processing H5 files"):
        h5_path = os.path.join(h5_dir, h5_filename)
        
        # Extract WSI ID from H5 filename (remove .h5 extension)
        wsi_id = os.path.splitext(h5_filename)[0]
        
        # Find corresponding WSI file
        wsi_path = find_wsi_file(wsi_dir, wsi_id)
        if not os.path.exists(wsi_path):
            print(f"Warning: Could not find WSI file for {wsi_id} at {wsi_path}, skipping")
            continue
        
        try:
            # Open WSI file
            wsi = openslide.OpenSlide(wsi_path)
            
            # Read coordinates and metadata from H5 file
            with h5py.File(h5_path, 'r') as hdf5_file:
                if 'coords' not in hdf5_file:
                    print(f"Warning: {h5_filename} missing 'coords' dataset, skipping")
                    wsi.close()
                    continue
                
                coords = hdf5_file['coords']
                
                # Get patch metadata from H5 attributes
                if 'patch_level' in coords.attrs:
                    patch_level = int(coords.attrs['patch_level'])
                else:
                    patch_level = 0  # Default to level 0
                
                if 'patch_size' in coords.attrs:
                    patch_size = int(coords.attrs['patch_size'])
                else:
                    patch_size = 256  # Default patch size
                
                num_patches = len(coords)
                print(f"Processing {num_patches} patches from {wsi_id} (level={patch_level}, size={patch_size})")
                
                # Process in batches for efficiency
                batch_size = 1000
                for batch_start in range(0, num_patches, batch_size):
                    batch_end = min(batch_start + batch_size, num_patches)
                    
                    # Prepare batch data as list of dictionaries
                    batch_data = []
                    
                    for idx in range(batch_start, batch_end):
                        coord = coords[idx]
                        coord_tuple = (int(coord[0]), int(coord[1]))
                        
                        # Extract patch from WSI using read_region
                        # read_region takes (x, y) at level 0, level, and size
                        patch_img = wsi.read_region(coord_tuple, patch_level, (patch_size, patch_size))
                        patch_img = patch_img.convert('RGB')
                        
                        # Convert PIL Image to binary (JPEG format)
                        img_bytes = io.BytesIO()
                        patch_img.save(img_bytes, format='JPEG')
                        img_binary = img_bytes.getvalue()
                        
                        # Create record
                        batch_data.append({
                            'wsi_id': wsi_id,
                            'patch_idx': idx,
                            'coord_x': coord_tuple[0],
                            'coord_y': coord_tuple[1],
                            'image': img_binary
                        })
                    
                    # Insert batch into LanceDB
                    table.add(batch_data)
            
            wsi.close()
        
        except Exception as e:
            print(f"Error processing {h5_filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"Successfully loaded patches into LanceDB table: {table_name}")


def load_from_lance(db_path, table_name, limit=None):
    """
    Load patches from LanceDB table for verification.
    
    Args:
        db_path: Path to the LanceDB database directory
        table_name: Name of the LanceDB table
        limit: Optional limit on number of patches to load
    """
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)
    
    # Get total count
    total = len(table)
    print(f"Total patches in table: {total}")
    
    # Load data
    if limit:
        df = table.head(limit).to_pandas()
    else:
        df = table.to_pandas()
    
    print(f"Loaded {len(df)} patches")
    print("\nSample data:")
    print(df.head())
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load WSI patches from H5 files into LanceDB")
    parser.add_argument(
        "--h5_dir",
        type=str,
        required=True,
        help="Directory containing H5 patch files"
    )
    parser.add_argument(
        "--wsi_dir",
        type=str,
        required=True,
        help="Directory containing WSI files"
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="./lancedb",
        help="Path to the LanceDB database directory (default: ./lancedb)"
    )
    parser.add_argument(
        "--table_name",
        type=str,
        default="wsi_patches",
        help="Name for the LanceDB table (default: wsi_patches)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Load and display sample data after creation"
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.h5_dir):
        print(f"Error: Directory {args.h5_dir} does not exist")
        exit(1)
    
    if not os.path.isdir(args.wsi_dir):
        print(f"Error: Directory {args.wsi_dir} does not exist")
        exit(1)
    
    start = time.time()
    process_h5_patches(args.h5_dir, args.wsi_dir, args.db_path, args.table_name)
    end = time.time()
    print(f"\nTime taken: {end - start:.2f} seconds")
    
    if args.verify:
        import pandas as pd
        print("\n" + "="*50)
        print("Verification:")
        print("="*50)
        load_from_lance(args.db_path, args.table_name, limit=10)
