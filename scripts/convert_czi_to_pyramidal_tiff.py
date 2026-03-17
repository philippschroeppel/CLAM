import argparse
import os
import glob

import cv2
import numpy as np
import tifffile
import aicspylibczi


def estimate_memory_gb(czi: aicspylibczi.CziFile) -> float:
    dims = czi.dims_shape()
    # Get spatial dimensions
    shape = {d['dimension']: d['length'] for entry in dims for d in [entry] if isinstance(entry, dict)}
    # dims_shape returns list of dicts; use total pixel count from bounding box
    bbox = czi.get_mosaic_bounding_box() if czi.is_mosaic() else czi.get_dims_shape()
    total_pixels = czi.size[0] * czi.size[1] if hasattr(czi, 'size') else None

    # Use bounding box for mosaic
    if czi.is_mosaic():
        bb = czi.get_mosaic_bounding_box()
        w, h = bb.w, bb.h
    else:
        ds = czi.dims_shape()
        # find X and Y
        w = next((d['X'] for d in ds if 'X' in d), None)
        h = next((d['Y'] for d in ds if 'Y' in d), None)
        if w is None or h is None:
            return -1

    # Assume 3 channels, uint16 (worst case)
    return w * h * 3 * 2 / 1e9


def read_image(czi: aicspylibczi.CziFile) -> np.ndarray:
    """Read CZI and return (H, W, C) uint8 or uint16 numpy array."""
    if czi.is_mosaic():
        data, _ = czi.read_mosaic(scale_factor=1.0)
    else:
        data, _ = czi.read_image()

    # data shape is typically (1, C, 1, H, W) or similar — squeeze extra dims
    data = np.squeeze(data)

    # Ensure (H, W, C) layout
    if data.ndim == 2:
        # Grayscale — keep as (H, W)
        pass
    elif data.ndim == 3:
        # Could be (C, H, W) or (H, W, C)
        if data.shape[0] <= 4 and data.shape[0] < data.shape[1]:
            # Likely (C, H, W) — transpose to (H, W, C)
            data = np.transpose(data, (1, 2, 0))

    return data


def build_pyramid_levels(image: np.ndarray, num_levels: int) -> list[np.ndarray]:
    levels = [image]
    for _ in range(num_levels):
        prev = levels[-1]
        h, w = prev.shape[:2]
        new_h, new_w = max(1, h // 2), max(1, w // 2)
        if prev.ndim == 3:
            downsampled = cv2.resize(prev, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            downsampled = cv2.resize(prev, (new_w, new_h), interpolation=cv2.INTER_AREA)
        levels.append(downsampled)
    return levels


def num_pyramid_levels(image: np.ndarray, tile_size: int) -> int:
    h, w = image.shape[:2]
    levels = 0
    while min(h, w) > tile_size:
        h, w = h // 2, w // 2
        levels += 1
    return levels


def convert(czi_path: str, output_path: str, tile_size: int):
    print(f'Reading {czi_path}...')
    czi = aicspylibczi.CziFile(czi_path)

    print(f'  Mosaic: {czi.is_mosaic()}')
    print(f'  Dims: {czi.dims}')

    mem_gb = estimate_memory_gb(czi)
    if mem_gb > 0:
        print(f'  Estimated memory: {mem_gb:.1f} GB')

    image = read_image(czi)
    print(f'  Image shape: {image.shape}, dtype: {image.dtype}')

    n_levels = num_pyramid_levels(image, tile_size)
    print(f'  Pyramid levels: {n_levels}')

    levels = build_pyramid_levels(image, n_levels)

    photometric = 'rgb' if image.ndim == 3 and image.shape[2] == 3 else 'minisblack'

    print(f'Writing {output_path}...')
    with tifffile.TiffWriter(output_path, bigtiff=True) as tif:
        options = dict(
            tile=(tile_size, tile_size),
            compression='lzw',
            photometric=photometric,
        )
        tif.write(levels[0], subifds=n_levels, **options)
        for level in levels[1:]:
            tif.write(level, subfiletype=1, **options)

    print(f'Done: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Convert CZI files to pyramidal TIFF.')
    parser.add_argument('--dir', required=True, help='Directory containing .czi files')
    parser.add_argument('--output_dir', help='Output directory (defaults to --dir)')
    parser.add_argument('--tile_size', type=int, default=256, help='Tile size in pixels (default: 256)')
    args = parser.parse_args()

    output_dir = args.output_dir or args.dir
    os.makedirs(output_dir, exist_ok=True)

    czi_files = glob.glob(os.path.join(args.dir, '*.czi')) + glob.glob(os.path.join(args.dir, '*.CZI'))
    czi_files = sorted(set(czi_files))

    if not czi_files:
        print(f'No .czi files found in {args.dir}')
        return

    for czi_path in czi_files:
        basename = os.path.splitext(os.path.basename(czi_path))[0]
        output_path = os.path.join(output_dir, basename + '_pyramid.tif')
        try:
            convert(czi_path, output_path, args.tile_size)
        except Exception as e:
            print(f'ERROR: {czi_path}: {e}')


if __name__ == '__main__':
    main()
