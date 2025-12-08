#!/usr/bin/env python3
"""
Download CAMELYON dataset from AWS S3.

This script downloads the CAMELYON (CAncer MEtastases in LYmph nOdes challeNge) 
dataset from the AWS Open Data Registry. The dataset contains whole slide images 
with corresponding annotations including tumor, stroma and tumor infiltrating lymphocytes.

Dataset information: https://registry.opendata.aws/camelyon/
S3 Bucket: s3://camelyon-dataset
"""

import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm


def download_file(s3_client, bucket_name, s3_key, local_path):
    """Download a single file from S3 with progress bar."""
    # Get file size for progress bar
    try:
        response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        file_size = response['ContentLength']
    except Exception as e:
        print(f"Warning: Could not get file size for {s3_key}: {e}")
        file_size = None

    # Create directory if it doesn't exist
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Download with progress bar
    try:
        with tqdm(total=file_size, unit='B', unit_scale=True, unit_divisor=1024, 
                  desc=os.path.basename(s3_key), leave=False) as pbar:
            def callback(bytes_amount):
                pbar.update(bytes_amount)
            
            s3_client.download_file(
                bucket_name, 
                s3_key, 
                str(local_path),
                Callback=callback
            )
        return True
    except Exception as e:
        print(f"Error downloading {s3_key}: {e}")
        return False


def list_s3_objects(s3_client, bucket_name, prefix=''):
    """List all objects in the S3 bucket."""
    objects = []
    paginator = s3_client.get_paginator('list_objects_v2')
    
    try:
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if 'Contents' in page:
                objects.extend(page['Contents'])
        return objects
    except Exception as e:
        print(f"Error listing objects in bucket: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description='Download CAMELYON dataset from AWS S3',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_camelyon.py /path/to/destination
  python download_camelyon.py ./data/camelyon
        """
    )
    parser.add_argument(
        'destination',
        type=str,
        help='Destination directory path where the dataset will be downloaded'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default='',
        help='Optional S3 prefix to download only specific subdirectories (default: download everything)'
    )
    
    args = parser.parse_args()
    
    # Validate and create destination directory
    dest_path = Path(args.destination).resolve()
    dest_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Destination directory: {dest_path}")
    print("Connecting to S3...")
    
    # Create S3 client with unsigned config (no credentials needed for public bucket)
    s3_client = boto3.client(
        's3',
        config=Config(signature_version=UNSIGNED),
        region_name='us-west-2'
    )
    
    bucket_name = 'camelyon-dataset'
    
    # List all objects in the bucket
    print("Listing files in S3 bucket...")
    objects = list_s3_objects(s3_client, bucket_name, prefix=args.prefix)
    
    if not objects:
        print("No files found in the bucket.")
        sys.exit(1)
    
    print(f"Found {len(objects)} files to download.")
    
    # Download all files
    successful = 0
    failed = 0
    
    for obj in tqdm(objects, desc="Downloading files"):
        s3_key = obj['Key']
        
        # Skip if it's a directory (ends with /)
        if s3_key.endswith('/'):
            continue
        
        # Create local file path
        local_file_path = dest_path / s3_key
        
        # Skip if file already exists
        if local_file_path.exists():
            file_size = obj.get('Size', 0)
            if local_file_path.stat().st_size == file_size:
                tqdm.write(f"Skipping {s3_key} (already exists)")
                successful += 1
                continue
        
        # Download the file
        if download_file(s3_client, bucket_name, s3_key, local_file_path):
            successful += 1
        else:
            failed += 1
    
    print(f"\nDownload complete!")
    print(f"Successfully downloaded: {successful} files")
    if failed > 0:
        print(f"Failed: {failed} files")
        sys.exit(1)


if __name__ == '__main__':
    main()

