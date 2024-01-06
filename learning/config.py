#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import argparse
import datetime

import torch

# helper function for passing None via CLI
def none_or_str(value):
    if value == 'None':
        return None
    return value

# parse args that correspond to configurations to be experimented on
parser = argparse.ArgumentParser()

# Experiment
parser.add_argument('--name',
                    help='Custom Experiment Name',
                    default='',
                    type=str)
parser.add_argument("--seed",
                    type=int,
                    default=0)
parser.add_argument('--log_dir',
                    default='save',
                    type=str)
parser.add_argument('--test_every',
                    default=1,
                    type=int)
parser.add_argument('--save_every',
                    default=100,
                    type=int)
parser.add_argument("--experiment_dir",
                    help="full path to experiment directory for loading files",
                    type=str)



# Dataset
parser.add_argument('--dataset',
                    default='C3',
                    choices=['C3', 'C3x', 'C3_neutral_lighting'], type=str)
parser.add_argument('--train_split',
                    help='Folder where the train split is located',
                    default='train',
                    type=str)
parser.add_argument('--eval_train_split',
                    default=None,
                    type=str)
parser.add_argument('--test_split',
                    help='Folder where the test split is located',
                    default='test',
                    type=str)
parser.add_argument('--val_split',
                    help='Folder where the validation split is located',
                    default='val',
                    type=str)
parser.add_argument('--data_root',
                    help='Folder where the dataset is located',
                    default='data', type=str)
parser.add_argument('--sampling_mode',
                    help='contrastive sampling mode',
                    default='randomwalk',
                    choices=['randomwalk', 'uniform'],
                    type=str)

parser.add_argument('--lr_decay_epochs',
                      type=list, action='store',
                      dest='lr_decay_epochs',
                      help='epochs where learning rate decays',
                      default=[0],)
 
# Network Configuration
parser.add_argument('--encoder', help='Network backbone',
                    default='LeNet5',
                    choices=['resnet18', 'LeNet5', 'AlexNet'], type=str)
parser.add_argument('--main_loss',
                    default='SimCLR',
                    choices=['SimCLR', 'VICReg', 'supervised', 'supervised_representation'], type=str)
parser.add_argument('--contrast',
                    default='time',
                    choices=['time', 'classic', 'combined', 'supervised', 'jitter', 'nocontrast', 'combined_jitter'], type=str)
parser.add_argument('--reg_loss',
                    default=None,
                    choices=[None, 'RELIC', 'Decorrelation'], type=none_or_str)
parser.add_argument('--lrate',
                    default=1e-3,
                    type=float)

parser.add_argument('--linear_nn',
                    dest='linear_nn',
                    action='store_true')
parser.add_argument('--no-linear_nn',
                    dest='linear_nn',
                    action='store_false')
parser.set_defaults(linear_nn=False)
parser.add_argument('--linear_nn_epochs',
                    default=200,
                    type=int)
parser.add_argument('--linear_nn_test_every',
                    default=10,
                    type=int)

parser.add_argument('--projectionhead',
                    dest='projectionhead',
                    action='store_true')
parser.add_argument('--no-projectionhead',
                    dest='projectionhead',
                    action='store_false')
parser.set_defaults(projectionhead=True)


parser.add_argument('--exhaustive_test',
                    dest='exhaustive_test',
                    action='store_true')
parser.add_argument('--no-exhaustive_test',
                    dest='exhaustive_test',
                    action='store_false')


# parser.add_argument('--save_model',
#                     dest='save_model',
#                     action='store_true')
# parser.add_argument('--no-save_model',
#                     dest='save_model',
#                     action='store_false')
# parser.set_defaults(save_model=True)

parser.add_argument('--save_embedding',
                    dest='save_embedding',
                    action='store_true')
parser.add_argument('--no-save_embedding',
                    dest='save_embedding',
                    action='store_false')
parser.set_defaults(save_embedding=False)
                    


# Hyperparameters
parser.add_argument('--num_workers',
                    default=4,
                    type=int)
parser.add_argument('--n_epochs',
                    default=100,
                    type=int)
parser.add_argument('--batch_size',
                    default=300,
                    help='Batch size of the SSL algorithm, in the train.py this also determines the batch size of the linear classifier',
                    type=int)
parser.add_argument('--feature_dim',
                    default=128,
                    type=int)
parser.add_argument('--hidden_dim',
                    default=128,
                    type=int)
parser.add_argument('--tau',
                    default=0.996,
                    type=float)
parser.add_argument('--crop_size',
                    default=32,
                    type=int)

parser.add_argument('--knn_batch_size',
                    default=256,
                    type=int)

parser.add_argument('--optimizer',
                    default="adam",
                    choices=["adam", "adamw"],
                    type=str)
parser.add_argument('--lrate_decay',
                    default=1.0,
                    type=float)
parser.add_argument('--decorr_weight',
                    default=0.4,
                    type=float)
parser.add_argument('--temperature',
                    default=1.,
                    type=float)
parser.add_argument('--similarity',
                    default='cosine',
                    choices=['cosine', 'RBF'], type=str)
parser.add_argument('--cosine_decay',
                    dest='cosine_decay',
                    action='store_true')
parser.set_defaults(cosine_decay=False)
parser.add_argument('--exp_decay',
                    dest='exp_decay',
                    action='store_true')
parser.set_defaults(exp_decay=False)
parser.add_argument('--weight_decay',
                    default=0.,
                    type=float)

# additional VicReg arguments
parser.add_argument("--sim-coeff", type=float, default=25.0,
                    help='Invariance regularization loss coefficient')
parser.add_argument("--std-coeff", type=float, default=25.0,
                    help='Variance regularization loss coefficient')
parser.add_argument("--cov-coeff", type=float, default=1.0,
                    help='Covariance regularization loss coefficient')


# color jittering
parser.add_argument("--jitter_brightness", type=float, default=0.8,
                    help='Value for brightness jitter in 0,1')
parser.add_argument("--jitter_contrast", type=float, default=0.8,
                    help='Value for contrast jitter in 0,1')
parser.add_argument("--jitter_saturation", type=float, default=0.8,
                    help='Value for saturation jitter in 0,1')
parser.add_argument("--jitter_hue", type=float, default=0.2,
                    help='Value for hue jitter in 0,0.5')
args, unknown = parser.parse_known_args()





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
