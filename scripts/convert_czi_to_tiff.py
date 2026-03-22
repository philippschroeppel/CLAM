import argparse
import glob
import os

import czifile
import tifffile


def convert_czi_to_tiff(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(input_dir, '*.czi')) +
                         glob.glob(os.path.join(input_dir, '*.CZI')))

    if not image_paths:
        print(f'No CZI files found in {input_dir}')
        return

    for image_path in image_paths:
        basename = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, basename + '.tif')

        if os.path.exists(output_path):
            print(f'Skipping {image_path} (already converted)')
            continue

        try:
            print(f'Converting {image_path}')
            img = czifile.imread(image_path)
            print(f'  shape: {img.shape}, dtype: {img.dtype}')

            tifffile.imwrite(output_path, img, compression='lzw')

            os.remove(image_path)
            print(f'  -> {output_path} (source deleted)')

        except Exception as e:
            print(f'ERROR: Failed to convert {image_path}: {e}')
            if os.path.exists(output_path):
                os.remove(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert CZI files to flat TIFF.')
    parser.add_argument('--dir', required=True, help='Directory containing CZI files')
    parser.add_argument('--output_dir', help='Output directory (defaults to --dir)')
    args = parser.parse_args()

    convert_czi_to_tiff(args.dir, args.output_dir or args.dir)
