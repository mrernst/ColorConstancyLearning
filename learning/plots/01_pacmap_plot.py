#!/usr/bin/python
# _____________________________________________________________________________


# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import os
import sys
import random
import datetime


import numpy as np
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, hsv_to_rgb

# configuration module
# -----

sys.path.append(os.path.dirname(sys.path[0]))
#import config
from config import args

# define manual random seed
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# custom libraries
# -----
from utils.general import save_model, load_model, save_args, AverageMeter, DotDict
from utils.configurator import get_augmentations, get_dataloaders, get_network, get_optimizer, get_scheduler, update_lr, get_loss
from utils.evaluation import fuse_representations, lls_fit, lls_eval, supervised_eval, knn_eval, wcss_bcss, get_pacmap, train_linear_classifier, train_linear_regressor, evaluate, log_to_tensorboard
from utils.visualization import ConfusionMatrix
from utils.networks import LinearClassifier, DoubleOutput



# custom preferences
# -----

device = (
	"cuda"
	if torch.cuda.is_available()
	else "mps"
	if torch.backends.mps.is_available()
	else "cpu"
)

# macOS does do its own multiprocessing and does not like it prescribed
args.num_workers = 0 if torch.backends.mps.is_available() else args.num_workers


data_properties_dict = {
	'C3': DotDict({'rgb_mean': (0.3621, 0.3644, 0.3635),
		   'rgb_std': (0.1456, 0.1479, 0.1477),
		   'classes': [str(c) for c in range(0,50)],
		   'n_classes': 50,
		   }),
	'C3x': DotDict({'rgb_mean': (0.3621, 0.3644, 0.3635),
	   'rgb_std': (0.1456, 0.1479, 0.1477),
	   'classes': [str(c) for c in range(0,150)],
	   'n_classes': 150,
	   }),
}

# temporal network
#args.exp_dir = "/Users/markus/Research/Code/ColorConstancyLearning/learning/save/03-01-24_17-31_001_basicrun_1_seed_1_C3_aug_time_SimCLR_reg_None/"

# random lighting network
args.exp_dir = "/Users/markus/Research/Code/ColorConstancyLearning/learning/save/05-01-24_14:23_002_basicrun_1_seed_1_C3_aug_time_SimCLR_reg_None"
# get the model you want to load
model_path = args.exp_dir + '/models/' + 'backbone_epoch_100.pt'
model = get_network(args, data_properties_dict).to(device)
load_model(model, model_path, device)
#model.load_state_dict(torch.load(model_path, map_location=torch.device('mps')))
model.eval()

# the DoubleOutput class is just needed because the fuse_representations function
# expects an output of (representation, projection) to work within the training loop
# and an encoder method that is only part of the network - I am faking this here with
# sequential modules to extract more representations

class layer4(torch.nn.Module):
	def __init__(self,):
		super().__init__()
		self.encoder = torch.nn.Sequential(
						model.layer1,
						model.layer2,
						model.layer3,
						model.projector,
		)
		self.out =  torch.nn.Sequential(
			self.encoder,
			DoubleOutput(),
		)

	def forward(self, x):
		return self.out(x)

l4 = layer4()
l4.eval()


class layer3(torch.nn.Module):
	def __init__(self,):
		super().__init__()
		self.encoder = torch.nn.Sequential(
						model.layer1,
						model.layer2,
						model.layer3,
		)
		self.out =  torch.nn.Sequential(
			self.encoder,
			DoubleOutput(),
		)

	def forward(self, x):
		return self.out(x)

l3 = layer3()
l3.eval()

class layer2(torch.nn.Module):
	def __init__(self,):
		super().__init__()
		self.encoder = torch.nn.Sequential(
						model.layer1,
						model.layer2,
						torch.nn.Flatten(),
		)
		self.out =  torch.nn.Sequential(
			self.encoder,
			DoubleOutput(),
		)

	def forward(self, x):
		return self.out(x)

l2 = layer2()
l2.eval()

class layer1(torch.nn.Module):
	def __init__(self,):
		super().__init__()
		self.encoder = torch.nn.Sequential(
						model.layer1,
						torch.nn.Flatten(),
		)
		self.out =  torch.nn.Sequential(
			self.encoder,
			DoubleOutput(),
		)

	def forward(self, x):
		return self.out(x)

l1 = layer1()
l1.eval()


class layer0(torch.nn.Module):
	def __init__(self,):
		super().__init__()
		#self.encoder = torch.nn.Sequential(
		#                torch.nn.Flatten(),
		#)
		self.encoder = torch.nn.Flatten()

		self.out =  torch.nn.Sequential(
			self.encoder,
			DoubleOutput(),
		)

	def forward(self, x):
		return self.out(x)

l0 = layer0()
l0.eval()


# for temporal data
args.data_root = "/Users/markus/Research/Code/ColorConstancyLearning/learning/data"
# get the dataloaders for the dataset
dataloader_train, dataloader_train_eval, dataloader_test, dataset_train, dataset_train_eval, dataset_test = get_dataloaders(
	args, data_properties_dict)


# custom libraries for plotting
# -----

import pacmap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.cm import get_cmap
import matplotlib.lines as mlines

from matplotlib.colors import ListedColormap, hsv_to_rgb
import seaborn as sns


# main program
# -----

# standard colormap for C3
hsl_colors = [(i * (360 / 50), 50, 100) for i in range(50)]

color_map = ListedColormap([hsv_to_rgb((hsl[0] / 360, hsl[1] / 100, hsl[2] / 100)) for hsl in hsl_colors])


fig, ax = plt.subplots(2, 5, figsize=(9,4))

e_names = ['Epoch$ = 0$', 'Epoch$ = 20$', 'Epoch$ = 40$', 'Epoch$ = 60$', 'Epoch$ = 80$']

for i,epoch in enumerate([0, 20, 40, 60, 80]):
	model_path = args.exp_dir + '/models/' + f'backbone_epoch_{epoch}.pt'
	model = get_network(args, data_properties_dict).to(device)
	if epoch == 0:
		pass
	else:
		load_model(model, model_path, device)
	####model.load_state_dict(torch.load(model_path, map_location=torch.device('mps')))
	model.eval()
	features_train_eval, labels_train_eval = fuse_representations(
		model, dataloader_train_eval, device=device)
	features_test, labels_test = fuse_representations(
		model, dataloader_test, device=device)
	
	labels = labels_test.cpu().numpy()
	embedding = pacmap.PaCMAP(n_components=2)
	X_transformed = embedding.fit_transform(features_test.cpu().numpy(), init="pca")
	
	ax[0,i].scatter(X_transformed[:, 0], X_transformed[:, 1], c=color_map(labels), s=.1, marker='s')
	ax[0,i].set_title(e_names[i], fontsize=14)
	ax[0,i].set_xticks([]), ax[0,i].set_yticks([])
	ax[0,i].spines['top'].set_visible(False)
	ax[0,i].spines['right'].set_visible(False)
	ax[0,i].spines['bottom'].set_visible(False)
	ax[0,i].spines['left'].set_visible(False)
	#ax[0,i].text(-0.08, 1.03, ['A','B','C','D','E','F'][i], transform=ax[i].transAxes, size=21, weight='bold')
ax[0,0].text(-0.25, 1.03, ['A','B','C','D','E','F'][0], transform=ax[0,0].transAxes, size=21, weight='bold')


l_names = [r'$\tilde{x}$', '$l_1$', '$l_2$', '$h$', '$z$']

for i,layer in enumerate([l0, l1, l2, l3, l4]):
	features_train_eval, labels_train_eval = fuse_representations(
		layer, dataloader_train_eval, device=device)
	features_test, labels_test = fuse_representations(
		layer, dataloader_test, device=device)
	labels = labels_test.cpu().numpy()
	embedding = pacmap.PaCMAP(n_components=2)
	X_transformed = embedding.fit_transform(features_test.cpu().numpy(), init="pca")
	
	ax[1,i].scatter(X_transformed[:, 0], X_transformed[:, 1], c=color_map(labels), s=.1, marker='s')
	ax[1,i].set_title(l_names[i], fontsize=14)
	ax[1,i].set_xticks([]), ax[1,i].set_yticks([])
	ax[1,i].spines['top'].set_visible(False)
	ax[1,i].spines['right'].set_visible(False)
	ax[1,i].spines['bottom'].set_visible(False)
	ax[1,i].spines['left'].set_visible(False)
	#ax[1,i].text(-0.08, 1.03, ['A','B','C','D','E','F'][i], transform=ax[1,i].transAxes, size=21, weight='bold')

ax[1,0].text(-0.25, 1.03, ['A','B','C','D','E','F'][1], transform=ax[1,0].transAxes, size=21, weight='bold')


plt.savefig('04_pacmap.pdf')
plt.show()

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
