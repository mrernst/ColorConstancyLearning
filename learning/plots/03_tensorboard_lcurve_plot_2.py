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


args.exp_dir = "/Users/markus/Research/Code/ColorConstancyLearning/learning/save/03-01-24_17-31_001_basicrun_1_seed_1_C3_aug_time_SimCLR_reg_None/"
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

import numpy as np
import pandas as pd

import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import collections
from matplotlib.patches import Rectangle
from scipy import stats

from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing import event_multiplexer


import itertools

# custom functions
# -----


def read_single_summary(path_to_tfevent, chist=0, img=0, audio=0, scalars=0,
						hist=0):
	ea = event_accumulator.EventAccumulator(path_to_tfevent, size_guidance={
			event_accumulator.COMPRESSED_HISTOGRAMS: chist,
			event_accumulator.IMAGES: img,
			event_accumulator.AUDIO: audio,
			event_accumulator.SCALARS: scalars,
			event_accumulator.HISTOGRAMS: hist,
											})
	ea.Reload()
	ea.Tags()
	return ea


def read_multiple_runs(path_to_project, chist=0, img=0, audio=0, scalars=0,
					   hist=0):
	# use with event_multiplexer (multiplexes different events together
	# useful for retraining I guess...)
	em = event_multiplexer.EventMultiplexer(size_guidance={
		event_accumulator.COMPRESSED_HISTOGRAMS: chist,
		event_accumulator.IMAGES: img,
		event_accumulator.AUDIO: audio,
		event_accumulator.SCALARS: scalars,
		event_accumulator.HISTOGRAMS: hist,
	})
	em.AddRunsFromDirectory(path_to_project)
	# load data
	em.Reload()
	return em


def convert_em_to_df(multiplexer):
	# this needs to be better and be able to cope with different scales
	# sort into training and testing/network
	df_dict = {}
	
	if len(multiplexer.Runs()) == 1:
		# figure out separate runs progressively
		entries = {}
		for run in multiplexer.Runs().keys():
			for tag in multiplexer.Runs()[run]["scalars"]:
				if tag.split('/')[0] not in entries.keys():
					entries[tag.split('/')[0]] = []
				entries[tag.split('/')[0]].append(tag)
		
		for run in entries:
			run_df = pd.DataFrame()
			for tag in entries[run]:
				tag_df = pd.DataFrame(multiplexer.Scalars(list(multiplexer.Runs().keys())[0], tag))
				tag_df = tag_df.drop(tag_df.columns[[0]], axis=1)
				run_df[tag] = tag_df.value
				run_df["step"] = tag_df.step
			df_dict[run] = run_df

	else:
		for run in multiplexer.Runs().keys():
			# create fresh empty dataframe
			run_df = pd.DataFrame()
			for tag in multiplexer.Runs()[run]["scalars"]:
				tag_df = pd.DataFrame(multiplexer.Scalars(run, tag))
				tag_df = tag_df.drop(tag_df.columns[[0]], axis=1)
				run_df[tag] = tag_df.value
				run_df["step"] = tag_df.step
			df_dict[run] = run_df

	return df_dict


from matplotlib.colors import ListedColormap, hsv_to_rgb
import seaborn as sns




# main program
# -----


em = read_multiple_runs('./save/')
df = convert_em_to_df(em)



#sns.set_theme()
sns.set_style("ticks")
sns.set_context("paper", font_scale=1.25, rc={"lines.linewidth": 2})
markers = ["o", "v", "s", "D", "o", "v", "s", "D", "o", "v", "s", "D"]
colors = sns.color_palette("colorblind", 12)
colors = sns.color_palette("tab10", 12)
colors = sns.color_palette()
colors = sns.color_palette("Set1", 12)


fig, axes = plt.subplots(1,2, figsize=(4,3), sharex=True, sharey=True)


# define axis parameters
ax = axes[0]

ax.grid(axis='y', zorder=0, alpha=0.5)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
ax.tick_params(
axis='x',          # changes apply to the x-axis
which='both',      # both major and minor ticks are affected
bottom=True,      # ticks along the bottom edge are off
top=False,         # ticks along the top edge are off
labelbottom=True) # labels along the bottom edge are off
ax.spines['left'].set_visible(False)

ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")

ax.text(-0.35, 1.05, ['A','B','C','D','E','F'][0], transform=ax.transAxes, size=21, weight='bold')
fig.subplots_adjust(bottom=.18)
fig.subplots_adjust(left=.14)

ax.set_ylim(0,1)
ax.set_xlim(0,100)


# 002
data = np.stack([
	df['05-01-24_14:23_002_basicrun_1_seed_1_C3_aug_time_SimCLR_reg_None']['accloss/test/class/accuracy'][:11],
	df['05-01-24_23:26_002_basicrun_2_seed_2_C3_aug_time_SimCLR_reg_None']['accloss/test/class/accuracy'][:11],
	df['06-01-24_08:15_002_basicrun_3_seed_3_C3_aug_time_SimCLR_reg_None']['accloss/test/class/accuracy'][:11],
	df['06-01-24_17:47_002_basicrun_4_seed_4_C3_aug_time_SimCLR_reg_None']['accloss/test/class/accuracy'][:11],
	df['07-01-24_11:49_002_basicrun_5_seed_5_C3_aug_time_SimCLR_reg_None']['accloss/test/class/accuracy'][:11],
])
mean = data.mean(0)
std = data.std(0)
#colors = sns.color_palette("ch:s=-.2,r=.6")
ax.plot(np.arange(-1,100,10), mean, label=r'${h}$', color=colors[3])
ax.fill_between(np.arange(-1,100,10),mean - std, mean + std, color=colors[3], alpha=0.3)


# 006
data = np.stack([
	df['05-01-24_14:27_006_supervisedbl_1_seed_1_C3_aug_nocontrast_supervised_reg_None']['accloss/test/class/accuracy'][:11],
	df['05-01-24_14:59_006_supervisedbl_2_seed_2_C3_aug_nocontrast_supervised_reg_None']['accloss/test/class/accuracy'][:11],
	df['05-01-24_15:30_006_supervisedbl_3_seed_3_C3_aug_nocontrast_supervised_reg_None']['accloss/test/class/accuracy'][:11],
	df['05-01-24_16:02_006_supervisedbl_4_seed_4_C3_aug_nocontrast_supervised_reg_None']['accloss/test/class/accuracy'][:11],
	df['05-01-24_16:32_006_supervisedbl_5_seed_5_C3_aug_nocontrast_supervised_reg_None']['accloss/test/class/accuracy'][:11],
])
mean = data.mean(0)
std = data.std(0)
#colors = sns.color_palette("ch:s=-.2,r=.6")
ax.plot(np.arange(-1,100,10), mean, label='superv.', color=colors[3], linestyle='--')
ax.fill_between(np.arange(-1,100,10),mean - std, mean + std, color=colors[3], alpha=0.3)

# plot linear pixel baseline (random data)
mean, std, stderr = 0.49376, 0.01407744, 0.01407744/np.sqrt(5)
ax.plot(np.arange(-1,100,10), np.repeat(mean, 11), linestyle=':', color=colors[3], label=r'${x}$')
ax.fill_between(np.arange(-1,100,10), np.repeat(mean - std, 11), np.repeat(mean + std, 11), color=colors[3], alpha=0.3)


ax.legend(frameon=False, fontsize=10, ncols=1)


# define axis parameters
ax = axes[1]

ax.grid(axis='y', zorder=0, alpha=0.5)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['bottom'].set_visible(False)
ax.tick_params(
axis='x',          # changes apply to the x-axis
which='both',      # both major and minor ticks are affected
bottom=True,      # ticks along the bottom edge are off
top=False,         # ticks along the top edge are off
labelbottom=True) # labels along the bottom edge are off
ax.spines['left'].set_visible(False)

ax.set_xlabel("Epoch")
#ax.set_ylabel("Accuracy")

#ax.text(-0.15, 1.05, ['A','B','C','D','E','F'][1], transform=ax.transAxes, size=21, weight='bold')
fig.subplots_adjust(bottom=.18)
fig.subplots_adjust(left=.14)

ax.set_ylim(0,1)
ax.set_xlim(0,100)





# 001
data = np.stack([
	df['03-01-24_17-31_001_basicrun_1_seed_1_C3_aug_time_SimCLR_reg_None']['accloss/test/class/accuracy'][:11],
	df['04-01-24_02-22_001_basicrun_2_seed_2_C3_aug_time_SimCLR_reg_None']['accloss/test/class/accuracy'][:11],
	df['04-01-24_10-58_001_basicrun_3_seed_3_C3_aug_time_SimCLR_reg_None']['accloss/test/class/accuracy'][:11],
	df['04-01-24_19-59_001_basicrun_4_seed_4_C3_aug_time_SimCLR_reg_None']['accloss/test/class/accuracy'][:11],
	df['05-01-24_04-50_001_basicrun_5_seed_5_C3_aug_time_SimCLR_reg_None']['accloss/test/class/accuracy'][:11],
])
mean = data.mean(0)
std = data.std(0)
#colors = sns.color_palette("ch:start=.2,rot=-.3")
ax.plot(np.arange(-1,100,10), mean, label='TT C3S', color=colors[4], linestyle='-')
ax.fill_between(np.arange(-1,100,10),mean - std, mean + std, color=colors[4], alpha=0.3)



# 005
data = np.stack([
	df['05-01-24_13:36_005_supervisedbl_1_seed_1_C3_aug_nocontrast_supervised_reg_None']['accloss/test/class/accuracy'][:11],
	df['05-01-24_14:07_005_supervisedbl_2_seed_2_C3_aug_nocontrast_supervised_reg_None']['accloss/test/class/accuracy'][:11],
	df['05-01-24_14:37_005_supervisedbl_3_seed_3_C3_aug_nocontrast_supervised_reg_None']['accloss/test/class/accuracy'][:11],
	df['05-01-24_15:06_005_supervisedbl_4_seed_4_C3_aug_nocontrast_supervised_reg_None']['accloss/test/class/accuracy'][:11],
	df['05-01-24_15:35_005_supervisedbl_5_seed_5_C3_aug_nocontrast_supervised_reg_None']['accloss/test/class/accuracy'][:11],
])
mean = data.mean(0)
std = data.std(0)
#colors = sns.color_palette("ch:start=.2,rot=-.3")
ax.plot(np.arange(-1,100,10), mean, label='sup. C3S', color=colors[4], linestyle='--')
ax.fill_between(np.arange(-1,100,10),mean - std, mean + std, color=colors[4], alpha=0.3)




# plot linear pixel baseline (temporal data)
mean, std, stderr = 0.58842,0.00650089,0.00650089/np.sqrt(5)
ax.plot(np.arange(-1,100,10), np.repeat(mean, 11), linestyle=':', color=colors[4], label=r'${x}$ C3S')
ax.fill_between(np.arange(-1,100,10), np.repeat(mean - std, 11), np.repeat(mean + std, 11), color=colors[4], alpha=0.3)




ax.legend(frameon=False, fontsize=10, ncols=1)

plt.show()

#__________________________________________________________

# Stick to 80 characters per line
# Use PEP8 Style
# Comment your code

# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
