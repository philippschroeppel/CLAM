import argparse
import pyvips as vips
import glob
import os


def convert_to_pyramid_tiff(dir_tiff, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Get all .tif files in the directory that don't have 'pyramid' in their name
    wsi = [f for f in glob.glob(os.path.join(dir_tiff, '*.tif')) if 'pyramid' not in f]
    imagenames = sorted(wsi)

    img_list = [os.path.basename(i) for i in imagenames]

    for wsi in img_list:
        basename = os.path.splitext(wsi)[0]

        tiffile = os.path.join(dir_tiff, basename + '.tif')
        pyramid_tiffile = os.path.join(output_dir, basename + 'pyramid' + '.tif')

        image = vips.Image.new_from_file(tiffile, access='sequential')
        image.tiffsave(pyramid_tiffile, compression='lzw', tile=True, tile_width=256, tile_height=256, pyramid=True, bigtiff=True)

        os.remove(tiffile)

        print(f'{tiffile} converted and deleted.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert .tif files to pyramidal tiff format.')
    parser.add_argument('--dir', required=True, help='Directory containing .tif files to convert')
    parser.add_argument('--output_dir', help='Output directory for pyramidal tiffs (defaults to --dir)')
    args = parser.parse_args()

    convert_to_pyramid_tiff(args.dir, args.output_dir or args.dir)

