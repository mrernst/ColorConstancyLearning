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


# parse arguments
# print relevant arguments to test whether this works
# data directory

print(args.experiment_dir)
# get the model you want to load

def load_model_and_layers(model_path):
	# get the model you want to load
	model = get_network(args, data_properties_dict).to(device)
	load_model(model, model_path, device)
	#model.load_state_dict(torch.load(model_path, map_location=torch.device('mps')))
	model.eval()
	
	# the DoubleOutput class is just needed because the fuse_representations function
	# expects an output of (representation, projection) to work within the training loop
	# and an encoder method that is only part of the network - I am faking this here with
	# sequential modules to extract more representations
	
	
	class layer5(torch.nn.Module):
		def __init__(self,):
			super().__init__()
			self.encoder = torch.nn.Sequential(
							model.layer1,
							model.layer2,
							model.layer3,
							model.layer4,
							model.projector,
			)
			self.out =  torch.nn.Sequential(
				self.encoder,
				DoubleOutput(),
			)
	
		def forward(self, x):
			return self.out(x)
	
	l5 = layer5()
	l5.eval()
	
	class layer4(torch.nn.Module):
		def __init__(self,):
			super().__init__()
			self.encoder = torch.nn.Sequential(
							model.layer1,
							model.layer2,
							model.layer3,
							model.layer4,
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
	
	#classifier.eval()
	return model,l5,l4,l3,l2,l1,l0


# for temporal data
# args.data_root = "/Users/markus/Research/Code/ColorConstancyLearning/learning/data"





# get the dataloaders for the dataset
dataloader_train, dataloader_train_eval, dataloader_test, dataset_train, dataset_train_eval, dataset_test = get_dataloaders(
	args, data_properties_dict)





# main program
# -----

path_to_tmp_evaluation = args.experiment_dir + "/temporal_hierarchical_eval/"
os.makedirs(path_to_tmp_evaluation, exist_ok=True)

s = 0
dataset_train_eval.label_by = 'object'
dataset_test.label_by = 'object'

dataloader_train_eval = DataLoader(
	dataset_train_eval, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
dataloader_test = DataLoader(
	dataset_test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)


# define an array that stores the accuracies
object_accuracy_array = np.zeros([1,5,6])
#for i,epoch in enumerate([1,10,20,30,40,50,60,70,80,90,100]):
# layer 0 done separately
model,l5,l4,l3,l2,l1,l0 = load_model_and_layers(model_path=args.experiment_dir + '/models/' + f'backbone_epoch_100.pt')
training_loss, training_acc, testing_loss, testing_acc = \
train_linear_classifier(train_dataloader=dataloader_train_eval, 
						test_dataloader=dataloader_test, 
						input_features=3072, 
						output_features=data_properties_dict[args.dataset].n_classes,
						model=l0, 
						learning_rate=1e-3, 
						epochs=args.linear_nn_epochs,
						timestep=0, 
						test_every=100, 
						confusion_matrix=None, 
						writer=None, 
						device=device)
print(training_loss, training_acc, testing_loss, testing_acc)
object_accuracy_array[s,:,0] = np.repeat(testing_acc, 5)

for i,epoch in enumerate([1,25,50,75,100]):
	# load the data into the model
	# refresh the layers
	model,l5,l4,l3,l2,l1,l0 = load_model_and_layers(model_path=args.experiment_dir + '/models/' + f'backbone_epoch_{epoch}.pt')
	for j, (layer, out_dim) in enumerate(zip([l0, l1, l2, l3, l4, l5],[3072, 1176, 400, 120, 84, 128])):
		if out_dim == 3072:
			pass
		else:
			training_loss, training_acc, testing_loss, testing_acc = \
			train_linear_classifier(train_dataloader=dataloader_train_eval, 
									test_dataloader=dataloader_test, 
									input_features=out_dim, 
									output_features=data_properties_dict[args.dataset].n_classes,
									model=layer, 
									learning_rate=1e-3, 
									epochs=args.linear_nn_epochs,
									timestep=epoch, 
									test_every=100, 
									confusion_matrix=None, 
									writer=None, 
									device=device)
			print(training_loss, training_acc, testing_loss, testing_acc)
			object_accuracy_array[s,i,j] = testing_acc

# save the array to a numpy file
np.save(path_to_tmp_evaluation + 'object_accuracy', object_accuracy_array)




dataset_train_eval.label_by = 'color'
dataset_test.label_by = 'color'

dataloader_train_eval = DataLoader(
	dataset_train_eval, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
dataloader_test = DataLoader(
	dataset_test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

# define an array that stores the accuracies
lighting_accuracy_array = np.zeros([1,5,6])

# layer 0 done separately
model,l5,l4,l3,l2,l1,l0 = load_model_and_layers(model_path=args.experiment_dir + '/models/' + f'backbone_epoch_100.pt')
training_loss, training_acc, testing_loss, testing_acc = \
train_linear_classifier(train_dataloader=dataloader_train_eval, 
						test_dataloader=dataloader_test, 
						input_features=3072, 
						output_features=24, 
						model=l0, 
						loss_fn=torch.nn.BCEWithLogitsLoss(),
						learning_rate=1e-3, 
						epochs=args.linear_nn_epochs,
						timestep=0, 
						test_every=100, 
						label_type='n_hot', 
						confusion_matrix=None, 
						writer=None, 
						device=device)
print(training_loss, training_acc.cpu(), testing_loss, testing_acc.cpu())
lighting_accuracy_array[s,:,0] = np.repeat(testing_acc.cpu(), 5)


#for i,epoch in enumerate([1,10,20,30,40,50,60,70,80,90,100]):
for i,epoch in enumerate([1,25,50,75,100]):
	# load the data into the model
	# refresh the layers
	model,l5,l4,l3,l2,l1,l0 = load_model_and_layers(model_path=args.experiment_dir + '/models/' + f'backbone_epoch_{epoch}.pt')    
	for j, (layer, out_dim) in enumerate(zip([l0, l1, l2, l3, l4, l5],[3072, 1176, 400, 120, 84, 128])):
		if out_dim == 3072:
			pass
		else:
			training_loss, training_acc, testing_loss, testing_acc = \
			train_linear_classifier(train_dataloader=dataloader_train_eval, 
									test_dataloader=dataloader_test, 
									input_features=out_dim, 
									output_features=24, 
									model=layer, 
									loss_fn=torch.nn.BCEWithLogitsLoss(),
									learning_rate=1e-3, 
									epochs=args.linear_nn_epochs,
									timestep=epoch, 
									test_every=100, 
									label_type='n_hot', 
									confusion_matrix=None, 
									writer=None, 
									device=device)
			print(training_loss, training_acc.cpu(), testing_loss, testing_acc.cpu())
			lighting_accuracy_array[s,i,j] = testing_acc.cpu()

# save the array to a numpy file
np.save(path_to_tmp_evaluation + 'lighting_accuracy.npy', lighting_accuracy_array)
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
