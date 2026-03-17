import argparse
import pyvips as vips
import glob
import os

# Formats that pyvips/libvips can typically read (with appropriate plugins installed)
# CZI requires libvips built with openslide >= 4.0 or bioformats support
DEFAULT_EXTENSIONS = ['.tif', '.tiff', '.czi', '.svs', '.ndpi', '.scn', '.mrxs', '.vms', '.vmu']


def convert_to_pyramid_tiff(input_dir, output_dir, extensions):
    os.makedirs(output_dir, exist_ok=True)

    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, f'*{ext}')))
        image_paths.extend(glob.glob(os.path.join(input_dir, f'*{ext.upper()}')))

    # Exclude files already converted
    image_paths = [p for p in image_paths if 'pyramid' not in os.path.basename(p)]
    image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f'No matching files found in {input_dir} for extensions: {extensions}')
        return

    for image_path in image_paths:
        basename = os.path.splitext(os.path.basename(image_path))[0]
        pyramid_tiffile = os.path.join(output_dir, basename + '_pyramid.tif')

        try:
            image = vips.Image.new_from_file(image_path, access='sequential')
            image.tiffsave(
                pyramid_tiffile,
                compression='lzw',
                tile=True,
                tile_width=256,
                tile_height=256,
                pyramid=True,
                bigtiff=True,
            )

            if output_dir != input_dir:
                os.remove(image_path)
                print(f'{image_path} converted to {pyramid_tiffile} and deleted.')
            else:
                os.remove(image_path)
                print(f'{image_path} converted to {pyramid_tiffile} and deleted.')

        except Exception as e:
            print(f'ERROR: Failed to convert {image_path}: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert whole-slide images to pyramidal TIFF format.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f'Supported extensions by default: {DEFAULT_EXTENSIONS}\n'
               'CZI support requires libvips with openslide >= 4.0 or bioformats.',
    )
    parser.add_argument('--dir', required=True, help='Directory containing images to convert')
    parser.add_argument('--output_dir', help='Output directory for pyramidal TIFFs (defaults to --dir)')
    parser.add_argument(
        '--extensions',
        nargs='+',
        default=DEFAULT_EXTENSIONS,
        help=f'File extensions to process (default: {DEFAULT_EXTENSIONS})',
    )
    args = parser.parse_args()

    convert_to_pyramid_tiff(args.dir, args.output_dir or args.dir, args.extensions)
