import torch
import os
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import math
from itertools import islice
import collections

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_MIL(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor([item[1] for item in batch])
	return [img, label]

def collate_tranformer(batch):
	img_s = torch.cat([item[0] for item in batch], dim = 0)
	coord_s = torch.cat([item[1] for item in batch], dim = 0)
	img_l = torch.cat([item[2] for item in batch], dim = 0)
	coord_l = torch.cat([item[3] for item in batch], dim = 0)
	label = torch.LongTensor([item[4] for item in batch])
	if len(batch[0]) >= 7:
		# Ragged mapping arrays stay one payload per slide; future model code can consume them.
		return [img_s, coord_s, img_l, coord_l, label, [item[5] for item in batch], [item[6] for item in batch]]
	return [img_s, coord_s, img_l, coord_l, label]


def get_simple_loader(dataset, batch_size=1, num_workers=1, mode='clam'):
	if(mode == 'transformer'):
		collate = collate_tranformer

	# Windows 上使用多进程 DataLoader 会触发主脚本再次导入，导致顶层代码重复执行。
	# 为了更稳定的行为，在 Windows 平台上强制使用单进程（num_workers=0）。
	if os.name == 'nt':
		worker_count = 0
	else:
		worker_count = num_workers if num_workers is not None else 4

	kwargs = {'num_workers': worker_count, 'pin_memory': False} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate, **kwargs)
	return loader 

def get_split_loader(split_dataset, training = False, testing = False, weighted = False, mode='clam'):
	"""
		return either the validation loader or training loader 
	"""
	if(mode == 'transformer'):
		collate = collate_tranformer

	# Windows 兼容：避免多进程引发主脚本重复执行
	if os.name == 'nt':
		worker_count = 0
	else:
		worker_count = 4

	kwargs = {'num_workers': worker_count} if device.type == "cuda" else {}
	if not testing:
		if training:
			if weighted:
				weights = make_weights_for_balanced_classes_split(split_dataset)
				loader = DataLoader(split_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate, **kwargs)
			else:
				loader = DataLoader(split_dataset, batch_size=1, sampler = RandomSampler(split_dataset), collate_fn = collate, **kwargs)
		else:
			loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate, **kwargs)
	
	else:
		n = len(split_dataset)
		if n <= 0:
			ids = []
		else:
			sample_n = max(1, int(n * 0.1))
			sample_n = min(sample_n, n)
			ids = np.random.choice(np.arange(n), sample_n, replace=False)
			ids = np.sort(ids)
		loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate, **kwargs)

	return loader

def get_optim(model, args):
	# Default: single-group optimizer
	base_lr = float(getattr(args, 'lr', 1e-4))
	weight_decay = float(getattr(args, 'reg', 0.0))

	# BiomedCLIP: parameter groups (prompt / text / others)
	use_param_groups = bool(
		getattr(args, 'model_type', None) == 'ViLa_MIL_BiomedCLIP'
		and hasattr(model, 'prompt_learner')
		and hasattr(model, 'text_encoder')
	)

	if use_param_groups:
		prompt_lr = getattr(args, 'prompt_lr', None)
		text_lr = getattr(args, 'text_lr', None)
		if prompt_lr is None:
			prompt_lr = min(max(base_lr * 10.0, 1e-4), 1e-3)
		if text_lr is None:
			text_lr = min(base_lr, 1e-5)

		prompt_params = [p for p in model.prompt_learner.parameters() if p.requires_grad]
		text_params = [p for p in model.text_encoder.parameters() if p.requires_grad]
		prompt_ids = {id(p) for p in prompt_params}
		text_ids = {id(p) for p in text_params}
		other_params = [
			p for p in model.parameters()
			if p.requires_grad and id(p) not in prompt_ids and id(p) not in text_ids
		]

		param_groups = []
		if len(other_params) > 0:
			param_groups.append({'params': other_params, 'lr': base_lr, 'weight_decay': weight_decay, 'name': 'other'})
		if len(prompt_params) > 0:
			param_groups.append({'params': prompt_params, 'lr': float(prompt_lr), 'weight_decay': weight_decay, 'name': 'prompt_learner'})
		if len(text_params) > 0:
			param_groups.append({'params': text_params, 'lr': float(text_lr), 'weight_decay': weight_decay, 'name': 'text_encoder'})

		if args.opt == "adam":
			optimizer = optim.Adam(param_groups)
		elif args.opt == 'sgd':
			optimizer = optim.SGD(param_groups, momentum=0.9)
		else:
			raise NotImplementedError
		return optimizer

	# Fallback for other models
	params = list(filter(lambda p: p.requires_grad, model.parameters()))
	if args.opt == "adam":
		optimizer = optim.Adam(params, lr=base_lr, weight_decay=weight_decay)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=weight_decay)
	else:
		raise NotImplementedError
	return optimizer

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None):
	indices = np.arange(samples).astype(int)
	
	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids)

		for c in range(len(val_num)):
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
			val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids

			remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
			all_val_ids.extend(val_ids)

			if custom_test_ids is None:  # sample test split
				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)
			# all_test_ids.extend(val_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)
			
			else:
				sample_num = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sampled_train_ids, all_val_ids, all_test_ids


def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error

def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))                                           
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)
