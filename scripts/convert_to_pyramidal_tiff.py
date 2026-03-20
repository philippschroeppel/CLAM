import argparse
import glob
import os

import numpy as np
import openslide
import tifffile

# Formats supported by OpenSlide 4.0 (CZI support added in 4.0)
DEFAULT_EXTENSIONS = ['.tif', '.tiff', '.czi', '.svs', '.ndpi', '.scn', '.mrxs', '.vms', '.vmu']


def convert_to_tiff(input_dir, output_dir, extensions):
    os.makedirs(output_dir, exist_ok=True)

    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, f'*{ext}')))
        image_paths.extend(glob.glob(os.path.join(input_dir, f'*{ext.upper()}')))

    image_paths = [p for p in image_paths if not p.endswith('.tif') or 'converted' not in os.path.basename(p)]
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f'No matching files found in {input_dir} for extensions: {extensions}')
        return

    for image_path in image_paths:
        basename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, basename + '.tif')

        if os.path.exists(output_path):
            print(f'Skipping {image_path} (already converted)')
            continue

        try:
            print(f'Converting {image_path}')
            slide = openslide.OpenSlide(image_path)
            w, h = slide.dimensions
            print(f'  dimensions: {w}x{h}')

            img = np.array(slide.read_region((0, 0), 0, (w, h)).convert('RGB'))
            slide.close()

            tifffile.imwrite(output_path, img, compression='lzw')

            os.remove(image_path)
            print(f'  -> {output_path} (source deleted)')

        except Exception as e:
            print(f'ERROR: Failed to convert {image_path}: {e}')
            if os.path.exists(output_path):
                os.remove(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert whole-slide images to TIFF using OpenSlide.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'Supported extensions by default: {DEFAULT_EXTENSIONS}\n'
               'Requires openslide-bin >= 4.0 for CZI support.',
    )
    parser.add_argument('--dir', required=True, help='Directory containing images to convert')
    parser.add_argument('--output_dir', help='Output directory (defaults to --dir)')
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=DEFAULT_EXTENSIONS,
        help=f'File extensions to process (default: {DEFAULT_EXTENSIONS})',
    )
    args = parser.parse_args()

    convert_to_tiff(args.dir, args.output_dir or args.dir, args.extensions)
