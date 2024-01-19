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


args.exp_dir = "/Users/markus/Research/Code/ColorConstancyLearning/learning/save/03-01-24_17:31_001_basicrun_1_seed_1_C3_aug_time_SimCLR_reg_None/"
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

path_to_tmp_evaluation = args.exp_dir + "/tmp/"
os.makedirs(path_to_tmp_evaluation, exist_ok=True)
# write config to a json file
save_args(path_to_tmp_evaluation, args.__dict__)

conf_mat = ConfusionMatrix(n_cls=data_properties_dict[args.dataset].n_classes)
# tensorboard writer
writer = SummaryWriter(log_dir=path_to_tmp_evaluation)

#for layer, out_dim in zip([l4, l3, l2, l1, l0],[128,84,400,1176,3072]):
for layer, out_dim in zip([l0],[3072]):
	train_loss, train_acc, test_loss, test_acc, = train_linear_classifier(
	dataloader_train_eval,
	dataloader_test,
	out_dim,
	data_properties_dict[args.dataset].n_classes,
	model=layer,
	confusion_matrix=conf_mat,
	epochs=200,#args.linear_nn_epochs,
	timestep=100,#epoch + 1,
	test_every=100,#args.linear_nn_test_every,
	writer=writer,
	device=device)
	print(train_loss, train_acc, test_loss, test_acc)


	#reload the last trained classifier (raw input)
	classifier_path = path_to_tmp_evaluation + '/models/'+'classifier_epoch_200.pt'
	classifier = LinearClassifier(num_features=3072)
	load_model(classifier, classifier_path, device)
	#model.load_state_dict(torch.load(model_path, map_location=torch.device('mps')))
	classifier.eval()
	
	
	# visualize weights in RGB
	weights = classifier.linear_out[1].weight.to('cpu').detach().numpy().reshape(50,3,32,32)
	weights -= weights.min()
	weights /= weights.max()
	print(weights.min(), weights.max())
	weights = weights[:,:,:,:]
	for i in range(50):
		plt.subplot(5, 10, i+1)
		weight = weights[i,:,:,:]
		weight = np.moveaxis(weight, 0, -1)
	
		plt.title(i)
		plt.imshow(weight)
		#plt.matshow(weight)
		#plt.clim(weights.min(), weights.max())  # as noted by @Eric Duminil, cmap='gray' makes the numbers stand out more
		frame1 = plt.gca()
		frame1.axes.get_xaxis().set_visible(False)
		frame1.axes.get_yaxis().set_visible(False)
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
