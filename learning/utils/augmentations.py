#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import torch
from torchvision.transforms import v2



# custom functions
# -----
def get_transformations(contrast_type, rgb_mean, rgb_std, crop_size):
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
			v2.ToImage(), 
			v2.ToDtype(torch.float32, scale=True),
			normalize,
		])
	
	val_transform = v2.Compose([
			v2.ToImage(),
			v2.ToDtype(torch.float32, scale=True),
			normalize,
		])
	
	if contrast_type == 'classic':
		# if classic use the TwoContrastTransform to create contrasts
		train_transform = TwoContrastTransform(train_transform)
	elif contrast_type == 'time':
		# if time, replace the created train_transform with val_transform
		train_transform = val_transform
		# or just return ToTensor() (what we do so far, but normalization might be good)
		# train_transform, val_transform = v2.ToTensor(), v2.ToTensor()
	elif contrast_type == 'nocontrast':
		train_transform = TwoContrastTransform(val_transform)
	elif contrast_type == 'supervised':
		train_transform = v2.Compose([
			v2.RandomResizedCrop(size=crop_size, scale=(0.2, 1.)),
			v2.RandomHorizontalFlip(),
			v2.ToImage(),
			v2.ToDtype(torch.float32, scale=True),
			normalize,
		])
		train_transform = TwoContrastTransform(train_transform)
	elif contrast_type == 'combined_jitter':
		train_transform = v2.Compose([
			v2.RandomApply([
				v2.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
			], p=0.8),
			v2.ToImage(),
			v2.ToDtype(torch.float32, scale=True),
			normalize,
		])
	elif contrast_type == 'combined_jitterplusgrayscale':
		train_transform = v2.Compose([
			v2.RandomApply([
				v2.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
			], p=0.8),
			v2.RandomGrayscale(p=0.2),
			v2.ToImage(),
			v2.ToDtype(torch.float32, scale=True),
			normalize,
			])


	# if augmentation_type == combined, just return
	return train_transform, val_transform

# custom classes
# -----

class TwoContrastTransform:
	"""
	Create two contrasts of the same image using the given
	torchvision transform
	"""
	def __init__(self, transform):
		self.transform = transform
	
	def __call__(self, x):
		return [self.transform(x), self.transform(x)]



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
