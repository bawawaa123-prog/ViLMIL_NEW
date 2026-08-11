from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
import os
import numpy as np
import time
import argparse
import pandas as pd
import math
import cv2
import openslide
from PIL import Image

SUPPORTED_WSI_SUFFIXES = ('.svs', '.ome.tif', '.ome.tiff', '.tif', '.tiff')
WSI_SEARCH_SUBDIRS = ('', 'benign', 'non_benign')


def normalize_slide_id(slide_name):
	"""Normalize slide id by stripping known whole-slide suffixes."""
	slide_name = str(slide_name).strip()
	lower_name = slide_name.lower()
	for suffix in ('.ome.tiff', '.ome.tif', '.svs', '.tiff', '.tif'):
		if lower_name.endswith(suffix):
			return slide_name[:len(slide_name) - len(suffix)]
	return os.path.splitext(slide_name)[0]


def _build_filename_candidates(slide_name):
	slide_name = str(slide_name).strip()
	lower_name = slide_name.lower()
	if any(lower_name.endswith(suffix) for suffix in SUPPORTED_WSI_SUFFIXES):
		return [slide_name]
	return [slide_name + suffix for suffix in SUPPORTED_WSI_SUFFIXES]


def resolve_wsi_path(source_root, slide_name):
	"""Resolve a slide path under source_root supporting svs/tif/ome.tif formats."""
	filename_candidates = _build_filename_candidates(slide_name)
	for subdir in WSI_SEARCH_SUBDIRS:
		search_root = source_root if subdir == '' else os.path.join(source_root, subdir)
		for filename in filename_candidates:
			path = os.path.join(search_root, filename)
			if os.path.isfile(path):
				return path
	return None


def build_thumbnail_via_tiles(wsi_path, target_size, source_dims=None, tile_size=4096):
	"""Build thumbnail using tiled read_region to avoid get_thumbnail decoder issues."""
	target_w, target_h = target_size
	if source_dims is None:
		probe_slide = openslide.OpenSlide(wsi_path)
		try:
			w0, h0 = probe_slide.dimensions
		finally:
			probe_slide.close()
	else:
		w0, h0 = source_dims

	canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
	scale_x = float(target_w) / float(w0)
	scale_y = float(target_h) / float(h0)
	total_tiles = math.ceil(h0 / tile_size) * math.ceil(w0 / tile_size)
	processed_tiles = 0

	tile_slide = openslide.OpenSlide(wsi_path)
	try:
		for y in range(0, h0, tile_size):
			read_h = min(tile_size, h0 - y)
			ty0 = int(y * scale_y)
			ty1 = int((y + read_h) * scale_y)
			if ty1 <= ty0:
				ty1 = min(target_h, ty0 + 1)

			for x in range(0, w0, tile_size):
				read_w = min(tile_size, w0 - x)
				tx0 = int(x * scale_x)
				tx1 = int((x + read_w) * scale_x)
				if tx1 <= tx0:
					tx1 = min(target_w, tx0 + 1)

				try:
					region = tile_slide.read_region((x, y), 0, (read_w, read_h)).convert('RGB')
					region_np = np.array(region)
				except Exception as e:
					print(f"[Warn] tile read failed at ({x}, {y}) size ({read_w}, {read_h}): {e}")
					processed_tiles += 1
					continue

				resized = cv2.resize(region_np, (tx1 - tx0, ty1 - ty0), interpolation=cv2.INTER_AREA)
				canvas[ty0:ty1, tx0:tx1] = resized
				processed_tiles += 1

				if processed_tiles == 1 or processed_tiles % 100 == 0 or processed_tiles == total_tiles:
					print(
						f"thumbnail fallback progress: {processed_tiles}/{total_tiles} "
						f"tiles for {os.path.basename(wsi_path)}"
					)
	finally:
		tile_slide.close()

	return Image.fromarray(canvas)


def get_associated_thumbnail(slide, max_side=None):
	"""Return associated thumbnail when available for single-level SVS/TIFF slides."""
	try:
		thumb = slide.associated_images.get('thumbnail')
	except Exception:
		thumb = None

	if thumb is None:
		return None

	thumb = thumb.convert('RGB')
	if max_side is not None:
		w, h = thumb.size
		if max(w, h) > max_side:
			ratio = float(max_side) / float(max(w, h))
			thumb = thumb.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.BILINEAR)
	return thumb


def _filter_contours_by_area(contours, hierarchy, filter_params):
	"""Filter contours and holes by configured area thresholds."""
	filtered = []
	hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
	all_holes = []

	for cont_idx in hierarchy_1:
		cont = contours[cont_idx]
		holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
		area = cv2.contourArea(cont)
		hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
		area = area - np.array(hole_areas).sum()
		if area <= 0:
			continue
		if area > filter_params['a_t']:
			filtered.append(cont_idx)
			all_holes.append(holes)

	foreground_contours = [contours[cont_idx] for cont_idx in filtered]
	hole_contours = []

	for hole_ids in all_holes:
		unfiltered_holes = [contours[idx] for idx in hole_ids]
		unfiltered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
		unfiltered_holes = unfiltered_holes[:filter_params['max_n_holes']]
		filtered_holes = []
		for hole in unfiltered_holes:
			if cv2.contourArea(hole) > filter_params['a_h']:
				filtered_holes.append(hole)
		hole_contours.append(filtered_holes)

	return foreground_contours, hole_contours


def segment_large_single_level(WSI_object, seg_params, filter_params, ref_patch_size=512, max_thumb_side=4096, wsi_path=None):
	"""Fallback segmentation for huge single-level TIFF/OME-TIFF slides using thumbnail."""
	start_time = time.time()

	w0, h0 = WSI_object.level_dim[0]
	ratio = min(float(max_thumb_side) / float(max(w0, h0)), 1.0)
	thumb_w = max(1, int(w0 * ratio))
	thumb_h = max(1, int(h0 * ratio))

	thumb_img = get_associated_thumbnail(WSI_object.wsi, max_side=max_thumb_side)
	if thumb_img is not None:
		thumb_w, thumb_h = thumb_img.size
		print(f"using associated thumbnail for segmentation: {thumb_w}x{thumb_h}")
	elif wsi_path is not None:
		print(f"associated thumbnail unavailable, building tiled thumbnail: {thumb_w}x{thumb_h}")
		thumb_img = build_thumbnail_via_tiles(wsi_path, (thumb_w, thumb_h), source_dims=(w0, h0), tile_size=4096)
	else:
		print(f"associated thumbnail unavailable, falling back to get_thumbnail: {thumb_w}x{thumb_h}")
		thumb_img = WSI_object.wsi.get_thumbnail((thumb_w, thumb_h)).convert('RGB')
	img = np.array(thumb_img.convert('RGB'))

	sthresh = int(seg_params.get('sthresh', 8))
	mthresh = int(seg_params.get('mthresh', 7))
	if mthresh < 1:
		mthresh = 1
	if mthresh % 2 == 0:
		mthresh += 1
	close = int(seg_params.get('close', 4))
	use_otsu = bool(seg_params.get('use_otsu', False))

	img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)
	if use_otsu:
		_, img_otsu = cv2.threshold(img_med, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
	else:
		_, img_otsu = cv2.threshold(img_med, sthresh, 255, cv2.THRESH_BINARY)

	if close > 0:
		kernel = np.ones((close, close), np.uint8)
		img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

	contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
	if hierarchy is None or len(contours) == 0:
		WSI_object.contours_tissue = []
		WSI_object.holes_tissue = []
		return WSI_object, time.time() - start_time

	hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
	scale_x = w0 / float(thumb_w)
	scale_y = h0 / float(thumb_h)

	scaled_ref_patch_area = int(ref_patch_size ** 2 / (scale_x * scale_y))
	local_filter_params = filter_params.copy()
	local_filter_params['a_t'] = local_filter_params['a_t'] * scaled_ref_patch_area
	local_filter_params['a_h'] = local_filter_params['a_h'] * scaled_ref_patch_area

	foreground_contours, hole_contours = _filter_contours_by_area(contours, hierarchy, local_filter_params)
	WSI_object.contours_tissue = WSI_object.scaleContourDim(foreground_contours, (scale_x, scale_y))
	WSI_object.holes_tissue = WSI_object.scaleHolesDim(hole_contours, (scale_x, scale_y))

	keep_ids = seg_params.get('keep_ids', [])
	exclude_ids = seg_params.get('exclude_ids', [])
	if len(keep_ids) > 0:
		contour_ids = set(keep_ids) - set(exclude_ids)
	else:
		contour_ids = set(np.arange(len(WSI_object.contours_tissue))) - set(exclude_ids)

	WSI_object.contours_tissue = [WSI_object.contours_tissue[i] for i in contour_ids]
	WSI_object.holes_tissue = [WSI_object.holes_tissue[i] for i in contour_ids]

	seg_time_elapsed = time.time() - start_time
	return WSI_object, seg_time_elapsed


def vis_wsi_preview_large_single_level(WSI_object, line_thickness=100, max_thumb_side=4096):
	"""Generate a lightweight tissue-mask preview for huge single-level slides."""
	w0, h0 = WSI_object.level_dim[0]
	ratio = min(float(max_thumb_side) / float(max(w0, h0)), 1.0)
	thumb_w = max(1, int(w0 * ratio))
	thumb_h = max(1, int(h0 * ratio))

	# For very large single-level TIFF, use white background to avoid additional WSI reads.
	img = np.full((thumb_h, thumb_w, 3), 255, dtype=np.uint8)
	only_mask = np.zeros(img.shape, dtype=np.uint8)

	scale = (thumb_w / float(w0), thumb_h / float(h0))
	line_thickness = max(1, int(line_thickness * ratio))

	if WSI_object.contours_tissue is not None and len(WSI_object.contours_tissue) > 0:
		scaled_contours = WSI_object.scaleContourDim(WSI_object.contours_tissue, scale)
		cv2.drawContours(img, scaled_contours, -1, (0, 255, 0), line_thickness, lineType=cv2.LINE_8)
		cv2.drawContours(only_mask, scaled_contours, -1, (0, 255, 0), line_thickness, lineType=cv2.LINE_8)

		if WSI_object.holes_tissue is not None:
			for holes in WSI_object.holes_tissue:
				scaled_holes = WSI_object.scaleContourDim(holes, scale)
				cv2.drawContours(img, scaled_holes, -1, (0, 0, 255), line_thickness, lineType=cv2.LINE_8)
				cv2.drawContours(only_mask, scaled_holes, -1, (0, 0, 255), line_thickness, lineType=cv2.LINE_8)

	return Image.fromarray(img), Image.fromarray(only_mask)

def stitching(file_path, wsi_object, downscale = 64):
	start = time.time()
	heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(255,255,255), alpha=-1, draw_grid=True)
	total_time = time.time() - start

	return heatmap, total_time

def segment(WSI_object, seg_params, filter_params):
	### Start Seg Timer
	start_time = time.time()

	# Segment
	WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

	### Stop Seg Timers
	seg_time_elapsed = time.time() - start_time
	return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
	### Start Patch Timer
	start_time = time.time()

	# Patch
	props = WSI_object.wsi.properties
	# Fallback chain for non-Aperio slides (e.g., generic TIFF/OME-TIFF).
	magnification = (
		props.get('aperio.AppMag')
		or props.get('openslide.objective-power')
		or '40'
	)
	kwargs['mag'] = str(magnification)
	file_path = WSI_object.process_contours(**kwargs)

	### Stop Patch Timer
	patch_time_elapsed = time.time() - start_time
	return file_path, patch_time_elapsed


def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, only_mask_save_dir, stitch_save_dir,
				  slide_name_file,
				  patch_size = 256, step_size = 256,
				  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
								'keep_ids': 'none', 'exclude_ids': 'none'},
				  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8},
				  vis_params = {'vis_level': -1, 'line_thickness': 500},
				  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
				  patch_level = 0,
				  use_default_params = False,
				  seg = False, save_mask = True,
				  stitch= False,
				  patch = False, auto_skip = True, process_list = None, uuid_name_file = None):

	all_data = np.array(pd.read_csv(uuid_name_file, header=None))
	slides = []
	id_names = {}
	slide_labels = {}
	for data in all_data:
		slides.append(data[1])
		id_names[data[1]] = data[0]
		slide_labels[data[1]] = data[2]  # Store the label (Adenocarcinoma/NonAdenocarcinoma)

	# Check for .svs files in appropriate subdirectories based on labels
	# Note: Based on actual file locations, NonAdenocarcinoma files are in 'benign' 
	# and Adenocarcinoma files are in 'non_benign'
	valid_slides = []
	slide_path_map = {}
	print(f"Checking {len(slides)} slides...")
	for slide in slides:
		file_path = resolve_wsi_path(source, slide)
		print(f"Looking for slide: {slide}")
		if file_path and os.path.isfile(file_path):
			valid_slides.append(slide)
			slide_path_map[slide] = file_path
			print(f"Found: {file_path}")
		else:
			print(f"Not found: {slide} (tried suffixes: {SUPPORTED_WSI_SUFFIXES})")
	
	print(f"Found {len(valid_slides)} valid slides out of {len(slides)} total slides")
	slides = valid_slides
	if process_list is None:
		df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)

	else:
		df = pd.read_csv(process_list)
		df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

	mask = df['process'] == 1
	process_stack = df[mask]

	total = len(process_stack)

	legacy_support = 'a' in df.keys()
	if legacy_support:
		print('detected legacy segmentation csv file, legacy support enabled')
		df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
						  'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
						  'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
						  'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
						  'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

	seg_times = 0.
	patch_times = 0.
	stitch_times = 0.

	for i in range(total):
		df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
		idx = process_stack.index[i]
		slide = process_stack.loc[idx, 'slide_id']
		print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
		print('processing {}'.format(slide))

		df.loc[idx, 'process'] = 0
		slide_id = normalize_slide_id(slide)

		if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
			print('{} already exist in destination location, skipped'.format(slide_id))
			print('{} already exist in destination location, skipped'.format(slide_id))
			df.loc[idx, 'status'] = 'already_exist'
			continue

		# Initialize WSI by resolving compatible suffixes.
		full_path = slide_path_map.get(slide, resolve_wsi_path(source, slide))
		if full_path is None:
			print(f"Unable to resolve slide path for: {slide}")
			df.loc[idx, 'status'] = 'missing_slide'
			continue
		WSI_object = WholeSlideImage(full_path)
		# Normalize output file naming to match CSV slide_id for downstream steps.
		WSI_object.name = slide_id

		if use_default_params:
			current_vis_params = vis_params.copy()
			current_filter_params = filter_params.copy()
			current_seg_params = seg_params.copy()
			current_patch_params = patch_params.copy()

		else:
			current_vis_params = {}
			current_filter_params = {}
			current_seg_params = {}
			current_patch_params = {}


			for key in vis_params.keys():
				if legacy_support and key == 'vis_level':
					df.loc[idx, key] = -1
				current_vis_params.update({key: df.loc[idx, key]})

			for key in filter_params.keys():
				if legacy_support and key == 'a_t':
					old_area = df.loc[idx, 'a']
					seg_level = df.loc[idx, 'seg_level']
					scale = WSI_object.level_downsamples[seg_level]
					adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
					current_filter_params.update({key: adjusted_area})
					df.loc[idx, key] = adjusted_area
				current_filter_params.update({key: df.loc[idx, key]})

			for key in seg_params.keys():
				if legacy_support and key == 'seg_level':
					df.loc[idx, key] = -1
				current_seg_params.update({key: df.loc[idx, key]})

			for key in patch_params.keys():
				current_patch_params.update({key: df.loc[idx, key]})

		if current_vis_params['vis_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_vis_params['vis_level'] = 0

			else:
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_vis_params['vis_level'] = best_level

		if current_seg_params['seg_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_seg_params['seg_level'] = 0

			else:
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_seg_params['seg_level'] = best_level

		keep_ids = str(current_seg_params['keep_ids'])
		if keep_ids != 'none' and len(keep_ids) > 0:
			str_ids = current_seg_params['keep_ids']
			current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['keep_ids'] = []

		exclude_ids = str(current_seg_params['exclude_ids'])
		if exclude_ids != 'none' and len(exclude_ids) > 0:
			str_ids = current_seg_params['exclude_ids']
			current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['exclude_ids'] = []

		w, h = WSI_object.level_dim[current_seg_params['seg_level']]
		is_large_single_level = len(WSI_object.level_dim) == 1 and (w * h > 1e8)
		if w * h > 1e8 and not is_large_single_level:
			print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
			df.loc[idx, 'status'] = 'failed_seg'
			continue
		if is_large_single_level:
			print('large single-level slide detected ({}x{}), using thumbnail-based segmentation fallback'.format(w, h))

		df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
		df.loc[idx, 'seg_level'] = current_seg_params['seg_level']


		seg_time_elapsed = -1
		if seg:
			if is_large_single_level:
				WSI_object, seg_time_elapsed = segment_large_single_level(
					WSI_object,
					current_seg_params,
					current_filter_params,
					wsi_path=full_path
				)
			else:
				WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params)

			if WSI_object.contours_tissue is None or len(WSI_object.contours_tissue) == 0:
				print('No valid tissue contours found after segmentation, skipping {}'.format(slide_id))
				df.loc[idx, 'status'] = 'failed_seg'
				continue

		if save_mask:
			try:
				if is_large_single_level:
					mask, only_mask = vis_wsi_preview_large_single_level(
						WSI_object,
						line_thickness=current_vis_params.get('line_thickness', 100),
						max_thumb_side=4096
					)
				else:
					mask, only_mask = WSI_object.visWSI(**current_vis_params)
				mask_path = os.path.join(mask_save_dir, slide_id +'.jpg')
				mask.save(mask_path)
				only_mask_path = os.path.join(only_mask_save_dir, slide_id +'.png')
				only_mask.save(only_mask_path)
			except Exception as e:
				print(f"[Warn] failed to save mask preview for {slide_id}: {e}")

		patch_time_elapsed = -1 # Default time
		if patch:
			current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size,
										 'save_path': patch_save_dir})
			file_path, patch_time_elapsed = patching(WSI_object = WSI_object,  **current_patch_params,)

		stitch_time_elapsed = -1
		if stitch:
			file_path = os.path.join(patch_save_dir, slide_id +'.h5')
			if os.path.isfile(file_path):
				try:
					heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
					stitch_path = os.path.join(stitch_save_dir, slide_id +'.jpg')
					# print(stitch_path)
					heatmap.save(stitch_path)
				except Image.DecompressionBombError as e:
					print(f"[Warn] stitching skipped for {slide_id}: {e}")
					stitch_time_elapsed = -1
				except Exception as e:
					print(f"[Warn] stitching failed for {slide_id}: {e}")
					stitch_time_elapsed = -1

		print("segmentation took {} seconds".format(seg_time_elapsed))
		print("patching took {} seconds".format(patch_time_elapsed))
		print("stitching took {} seconds".format(stitch_time_elapsed))
		df.loc[idx, 'status'] = 'processed'

		seg_times += seg_time_elapsed
		patch_times += patch_time_elapsed
		stitch_times += stitch_time_elapsed

	seg_times /= total
	patch_times /= total
	stitch_times /= total

	df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
	print("average segmentation time in s per slide: {}".format(seg_times))
	print("average patching time in s per slide: {}".format(patch_times))
	print("average stiching time in s per slide: {}".format(stitch_times))

	return seg_times, patch_times

parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type = str,
					help='path to folder containing raw wsi image files')
parser.add_argument('--step_size', type = int, default=256,
					help='step_size')
parser.add_argument('--patch_size', type = int, default=256,
					help='patch_size')
parser.add_argument('--patch', default=False, action='store_true')
parser.add_argument('--seg', default=False, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=False, action='store_false')
parser.add_argument('--save_dir', type = str,
					help='directory to save processed data')
parser.add_argument('--preset', default=None, type=str,
					help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=0,
					help='downsample level at which to patch')
parser.add_argument('--process_list',  type = str, default=None,
					help='name of list of images to process with parameters (.csv)')

parser.add_argument('--slide_name_file', type=str,
					help='a file stored all slides name needed in this project')
parser.add_argument('--uuid_name_file', type=str,
					help='a file stored all slides info')

if __name__ == '__main__':
	args = parser.parse_args()

	patch_save_dir = os.path.join(args.save_dir, 'patches_' + str(args.patch_size))
	mask_save_dir = os.path.join(args.save_dir, 'masks')
	stitch_save_dir = os.path.join(args.save_dir, 'graph_' + str(args.patch_size))
	only_mask_save_dir = os.path.join(args.save_dir, 'only_masks')

	if args.process_list:
		process_list = os.path.join(args.save_dir, args.process_list)

	else:
		process_list = None

	print('source: ', args.source)
	print('patch_save_dir: ', patch_save_dir)
	print('mask_save_dir: ', mask_save_dir)
	print('stitch_save_dir: ', stitch_save_dir)

	directories = {'source': args.source,
				   'save_dir': args.save_dir,
				   'patch_save_dir': patch_save_dir,
				   'mask_save_dir' : mask_save_dir,
				   'only_mask_save_dir' : only_mask_save_dir,
				   'stitch_save_dir': stitch_save_dir}

	for key, val in directories.items():
		print("{} : {}".format(key, val))
		if key not in ['source']:
			os.makedirs(val, exist_ok=True)

	seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'}
	filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
	vis_params = {'vis_level': -1, 'line_thickness': 250}
	patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	if args.preset:
		preset_df = pd.read_csv(os.path.join('presets', args.preset))
		for key in seg_params.keys():
			seg_params[key] = preset_df.loc[0, key]

		for key in filter_params.keys():
			filter_params[key] = preset_df.loc[0, key]

		for key in vis_params.keys():
			vis_params[key] = preset_df.loc[0, key]

		for key in patch_params.keys():
			patch_params[key] = preset_df.loc[0, key]

	parameters = {'seg_params': seg_params,
				  'filter_params': filter_params,
				  'patch_params': patch_params,
				  'vis_params': vis_params}

	print(parameters)

	seg_times, patch_times = seg_and_patch(**directories, **parameters,
										   slide_name_file=args.slide_name_file,
										   patch_size = args.patch_size, step_size=args.step_size,
										   seg = args.seg,  use_default_params=False, save_mask = True,
										   stitch= args.stitch,
										   patch_level=args.patch_level, patch = args.patch,
										   process_list = process_list, auto_skip=args.no_auto_skip, uuid_name_file=args.uuid_name_file)
