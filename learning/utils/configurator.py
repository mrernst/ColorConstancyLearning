#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, StepLR

import os

from torchvision import utils
from torchvision.transforms import v2

from utils.general import TwoContrastTransform, DotDict
from utils.losses import RELIC_Loss, VICReg_Loss, SimCLR_Loss, Decorrelation_Loss
from utils.networks import ResNet18, LeNet5, AlexNet, MLPHead
from utils.datasets import SimpleTimeContrastiveDataset



# # use the pytorch 2 transforms if you are on pt2
# if int(torch.__version__[0]) == 2:
# 	from torchvision.transforms import v2 as transforms
	# for now this is fine, but ToTensor is being deprecated


# custom functions
# -----

def get_augmentations(contrast_type:str, rgb_mean:float, rgb_std:float, crop_size:float):
	"""
	contrast_type: str 'classic', 'cltt', else
	rgb_mean: tuple of float (r, g, b)
	rgb_std: tuple of float (r, g, b)
	crop_size: int, pixels
	"""
	
	# setup for case contrast_type == 'combined'
	normalize = v2.Normalize(mean=rgb_mean, std=rgb_std)
	
	s = 1.0
	train_transform = v2.Compose([
			v2.RandomResizedCrop(size=crop_size, scale=(0.2, 1.)),
			v2.RandomHorizontalFlip(),
			v2.RandomApply([
				v2.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
			], p=0.8),
			v2.RandomGrayscale(p=0.2),
			v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
			normalize,
		])
	
	val_transform = v2.Compose([
			v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
			normalize,
		])
	
	test_transform = val_transform
	
	if contrast_type == 'classic':
		# if classic use the TwoContrastTransform to create contrasts
		train_transform = TwoContrastTransform(train_transform)
	elif contrast_type == 'time':
		# if time, replace the created train_transform with val_transform
		train_transform = val_transform
		# or just return ToTensor() (what we do so far, but normalization might be good)
		# train_transform, val_transform = v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.ToTensor()
	elif contrast_type == 'nocontrast':
		train_transform = TwoContrastTransform(val_transform)
	elif contrast_type == 'jitter':
		# TODO: hyperparameter tuning for the hue jitter variable
		train_transform = v2.Compose([
			v2.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s), #0.2*s is default
			v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
			normalize,
		])
		#val_transform = train_transform
		train_transform = TwoContrastTransform(train_transform)
	elif contrast_type == 'supervised':
		train_transform = v2.Compose([
			v2.RandomResizedCrop(size=crop_size, scale=(0.2, 1.)),
			v2.RandomHorizontalFlip(),
			v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
			normalize,
		])
		train_transform = TwoContrastTransform(train_transform)
	elif contrast_type == 'combined_jitter':
		train_transform = v2.Compose([
			v2.RandomApply([
				v2.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
			], p=0.8),
			v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
			normalize,
		])
	elif contrast_type == 'combined_jitterplusgrayscale':
		train_transform = v2.Compose([
			v2.RandomApply([
				v2.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
			], p=0.8),
			v2.RandomGrayscale(p=0.2),
			v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
			normalize,
			])

	return train_transform, val_transform, test_transform



def get_dataloaders(args, data_properties_dict):
	
	# get transformations for validation and for training
	train_transform, val_transform, test_transform = get_augmentations(
		contrast_type=args.contrast,
		rgb_mean=data_properties_dict[args.dataset].rgb_mean,
		rgb_std=data_properties_dict[args.dataset].rgb_std,
		crop_size=args.crop_size)
	
	train_root = os.path.expanduser(args.data_root) + f'/{args.dataset}{args.train_split}' if args.train_split[0] == '_' else os.path.expanduser(args.data_root) + f'/{args.dataset}/{args.train_split}'
	print(f"[INFO:] Training set at '{train_root}'")
	dataset_train = SimpleTimeContrastiveDataset(
		root=train_root,
		transform=train_transform,
		contrastive=True if (args.contrast == 'time' or
							 'combined' in args.contrast) else False,)
	
	dataloader_train = DataLoader(
		dataset_train, batch_size=args.batch_size,
								  num_workers=args.num_workers, shuffle=True, drop_last=True)
	# if there is a eval_train_split use it, if not go for the train split
	if args.eval_train_split:
		train_eval_root = os.path.expanduser(args.data_root) + f'/{args.dataset}{args.eval_train_split}' if args.eval_train_split[0] == '_' else os.path.expanduser(args.data_root) + f'/{args.dataset}/{args.eval_train_split}'
	else:
		train_eval_root = os.path.expanduser(args.data_root) + f'/{args.dataset}{args.train_split}' if args.train_split[0] == '_' else os.path.expanduser(args.data_root) + f'/{args.dataset}/{args.train_split}'
	print(f"[INFO:] Evaluation Training set at '{train_eval_root}'")
	dataset_train_eval = SimpleTimeContrastiveDataset(
		root=train_eval_root,
		transform=val_transform,
		contrastive=False
	)
	
	dataloader_train_eval = DataLoader(
		dataset_train_eval, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
	
	
	test_root = os.path.expanduser(args.data_root) + f'/{args.dataset}{args.test_split}' if args.test_split[0] == '_' else os.path.expanduser(args.data_root) + f'/{args.dataset}/{args.test_split}'
	print(f"[INFO:] Test set at '{test_root}'")
	dataset_test = SimpleTimeContrastiveDataset(
		root=test_root,
		transform=test_transform,
		contrastive=False
	)
	dataloader_test = DataLoader(
		dataset_test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
	
	
	return dataloader_train, dataloader_train_eval, dataloader_test

def get_network(args, data_properties_dict):
		
	if args.encoder == 'resnet18':
		network = ResNet18(num_classes=data_properties_dict[args.dataset].n_classes)
	elif args.encoder == 'LeNet5':
		network = LeNet5(num_classes=data_properties_dict[args.dataset].n_classes)
	elif args.encoder == 'AlexNet':
		network = AlexNet(num_classes=data_properties_dict[args.dataset].n_classes)
	else:
		raise NotImplementedError(
			'[INFO] Specified Encoder is not implemented')	

	return network

def get_optimizer(model, args):
	
	optimizer = torch.optim.AdamW(
	model.parameters(), lr=args.lrate, weight_decay=args.weight_decay)
	
	return optimizer


# TODO: convert this into a Loss class with a method get_loss to avoid
# reloading the dicts every time this function is called
def get_loss(projection:torch.Tensor, pair:torch.Tensor, label:torch.Tensor, args):
	
	sim_func_dict = {
		'cosine': lambda x, x_pair: F.cosine_similarity(x.unsqueeze(1), x_pair.unsqueeze(0), dim=2),
		'RBF': lambda x, x_pair: -torch.cdist(x, x_pair)
	}
	
	main_loss_dict = {
		'SimCLR': SimCLR_Loss(sim_func_dict[args.similarity], args.batch_size, args.temperature),
		'VICReg': VICReg_Loss(),
		'supervised': lambda x, x_pair, labels: F.cross_entropy(x, labels),
		'supervised_representation': lambda x, x_pair, labels: F.cross_entropy(x, labels),
	}
	reg_loss_dict = {
		'RELIC': RELIC_Loss(sim_func_dict[args.similarity]),
		'Decorrelation': Decorrelation_Loss(args.hidden_dim)
	}

	loss = main_loss_dict[args.main_loss](projection, pair, label)
	if args.reg_loss:
		if args.reg_loss == 'Decorrelation':
			representation = representation.split(
				projection.shape[0]//2)[0].T
			loss += args.decorr_weight * \
				reg_loss_dict[args.reg_loss](representation, pair)
		else:
			loss += reg_loss_dict[args.reg_loss](projection, pair)
	
	return loss

def get_scheduler(optimizer, args):
	if args.cosine_decay:
		scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs,
									  eta_min=args.lrate * (args.lrate_decay ** 3))
	elif args.exp_decay:
		scheduler = ExponentialLR(optimizer, 1.0)
	else:
		scheduler = StepLR(optimizer, 10, args.lrate_decay)
	
	return scheduler

def update_lr(scheduler, epoch, args):
	lr_decay_steps = torch.sum(epoch > torch.Tensor(args.lr_decay_epochs))
	if (lr_decay_steps > 0 and not (args.cosine_decay or args.exp_decay)):
		scheduler.gamma = args.lrate_decay ** lr_decay_steps
	scheduler.step()
	pass




# ----------------
# main program
# ----------------

if __name__ == "__main__":
	pass


# _____________________________________________________________________________

# Stick to 80 characters per line
# Use PEP8 Style
# Comment your code

# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
