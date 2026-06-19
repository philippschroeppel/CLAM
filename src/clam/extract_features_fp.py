import time
import os
import argparse
import pdb
from functools import partial

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

import numpy as np

from dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models import get_encoder
from wsi_core.output_adapters import build_coord_input, build_feature_output
from wsi_core.slide_io import open_slide

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def compute_w_loader(slide_id, feature_output, loader, model, verbose = 0):
	"""
	args:
		feature_output: destination adapter for computed features
		model: pytorch model
		verbose: level of feedback
	"""
	if verbose > 0:
		print(f'processing a total of {len(loader)} batches'.format(len(loader)))

	mode = 'w'
	for count, data in enumerate(tqdm(loader)):
		with torch.inference_mode():	
			batch = data['img']
			coords = data['coord'].numpy().astype(np.int32)
			batch = batch.to(device, non_blocking=True)
			
			features = model(batch)
			features = features.cpu().numpy().astype(np.float32)
			patch_indices = data['patch_idx'].numpy().astype(np.int64)

			feature_output.write_batch(slide_id, features, coords, patch_indices, mode=mode)
			mode = 'a'
	
	return feature_output.finalize_slide(slide_id)


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'uni_v1', 'conch_v1'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--target_patch_size', type=int, default=224)
parser.add_argument('--slide_backend', choices=['auto', 'openslide', 'czi'], default='auto')
parser.add_argument('--coords_adapter', choices=['hdf5', 'lance'], default='hdf5')
parser.add_argument('--coords_lance_db_path', type=str, default=None)
parser.add_argument('--coords_lance_table_name', type=str, default='wsi_patch_coords')
parser.add_argument('--feature_output_adapter', choices=['hdf5', 'lance'], default='hdf5')
parser.add_argument('--feature_lance_db_path', type=str, default=None)
parser.add_argument('--feature_lance_table_name', type=str, default='wsi_patch_embeddings')
args = parser.parse_args()


if __name__ == '__main__':
	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	coord_input = build_coord_input(
		args.coords_adapter,
		data_h5_dir=args.data_h5_dir,
		lance_db_path=args.coords_lance_db_path,
		lance_table_name=args.coords_lance_table_name,
	)
	feature_output = build_feature_output(
		args.feature_output_adapter,
		args.feat_dir,
		lance_db_path=args.feature_lance_db_path,
		lance_table_name=args.feature_lance_table_name,
		model_name=args.model_name,
	)

	model, img_transforms = get_encoder(args.model_name, target_img_size=args.target_patch_size)
			
	_ = model.eval()
	model = model.to(device)
	total = len(bags_dataset)

	loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

	for bag_candidate_idx in tqdm(range(total)):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)
		coords, coord_attrs = coord_input.load(slide_id)

		if not args.no_auto_skip and feature_output.exists(slide_id, expected_count=len(coords)):
			print('skipped {}'.format(slide_id))
			continue 

		time_start = time.time()
		wsi = open_slide(slide_file_path, backend=args.slide_backend)
		dataset = Whole_Slide_Bag_FP(wsi=wsi,
									 img_transforms=img_transforms,
									 coords=coords,
									 coord_attrs=coord_attrs)

		loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)
		output_file_path = compute_w_loader(slide_id, feature_output, loader = loader, model = model, verbose = 1)

		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

