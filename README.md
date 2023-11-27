# Towards Self-Supervised Learning of Color Constancy

<p align="center">
  <img src="https://github.com/mrernst/colorconstancylearning/blob/main/img/header.png" width="500">


This is the codebase used for the ICDL publication "Towards Self-Supervised Learning of Color Constancy" [1]. If you make use of this code please cite as follows:

[1] M. R. Ernst, F. M. López, R. W. Fleming, A. Aubret and J. Triesch, Towards Self-Supervised Learning of Color Constancy, In Proceedings of the International Conference on Development and Learning. ICDL 2024 (pp. XXX-XXX). IEEE. The paper can found at [arxiv](https://arxiv.org/).

### Overleaf Doc for the ICDL paper
[Link](https://www.overleaf.com/project/64e861676c537ba8d586d233)

### Google Doc for collaboration and ideas
[Link](https://docs.google.com/document/d/1__e8eMz4xCEDY_x3UzuMvk4r9OnP_nCHj6TaeB6uOYs/edit?usp=sharing)


## Getting started with the code
 Clone the repository from here
* Make sure you have all the dependencies installed, see also requirements.txt
* Generate the corresponding color-constancy dataset using Blender and [BlenderProc]()
* Start an experiment on your local machine with python3 main/train.py


### Prerequisites

* [numpy](http://www.numpy.org/)
* [pytorch](https://www.pytorch.org/)
* [matplotlib](https://matplotlib.org/)
* [tensorboard](https://tensorflow.org/)
* [pandas](https://pandas.pydata.org)
* [tqdm](https://pypi.org/project/tqdm/)


### Repository structure

```bash
.
├── codetemplate                       # Template for collaboration purposes
├── config.py                          # Configuration for experiment parameters
├── data                               
│   └── XXXX                 	       # dataset (not included, please generate)
├── img
├── LICENSE                            # MIT License
├── main
│   └── train.py                       # main training file     		                 		    
├── README.md                          # ReadMe File
├── requirements.txt                   # conda/pip requirements
├── utils
│   ├── augmentations.py               # augmentations for standard SimCLR
│   ├── datasets.py                    # dataloading and sampling
│   ├── evaluation.py                  # evaluation methods and analytics
│   ├── general.py                     # general IO utilities
│   ├── losses.py                      # definition of loss functions
└── └── networks.py                    # network backbone, e.g. ResNet

```

### Installation guide

#### Forking the repository

Fork a copy of this repository onto your own GitHub account and `clone` your fork of the repository into your computer, inside your favorite folder, using:

`git clone "PATH_TO_FORKED_REPOSITORY"`

#### Setting up the environment

Install [Python 3.9](https://www.python.org/downloads/release/python-395/) and the [conda package manager](https://conda.io/miniconda.html) (use miniconda). Navigate to the project directory inside a terminal and create a virtual environment (replace <ENVIRONMENT_NAME>, for example, with `CORe50Env`) and install the [required packages](requirements.txt):

`conda create -n <ENVIRONMENT_NAME> --file requirements.txt python=3.9`

Activate the virtual environment:

`source activate <ENVIRONMENT_NAME>`


#### Generating the dataset


#### Starting an experiment
Starting an experiment is straight forward. Just execute the script from the main level and specify your options via the command line.

##### Example A) A run with the CORe50Env
```bash
python3 main/train.py \
	--name CEnv_Exp_0 \                             # specify experiment name
	--data_root './data/' \                         # specify where you put the CORe50 dataset
	--n_fix 0.95 \                                  # specify N_o as float probability [0,1]
	--n_fix_per_session 0.95 \                      # specify N_s as float probability [0,1]
	--contrast 'time' \                             # choose 'time' or 'combined' for -TT or TT+
	--view_sampling randomwalk \                    # choose 'randomwalk' or 'uniform'
	--test_every 10 \                               # test every 10 epochs
	--train_split train_alt_0 \                     # choose the splits for cross-validation (k in range(5))
	--test_split test_alt_0 \
	--val_split val_alt_0 \

```

##### Example B) A comparison run with standard SimCLR
```bash
python3 main/train.py \
	--name SimCLR_Exp_0 \                           # specify experiment name
	--data_root './data/' \                         # specify where you put the CORe50 dataset
	--contrast 'classic' \                          # choose 'classic' for SimCLR type contrasts
	--test_every 10 \                               # test every 10 epochs
	--train_split train_alt_0 \                     # choose the splits for cross-validation (k in range(5))
	--test_split test_alt_0 \
	--val_split val_alt_0 \

```

##### Example C) A comparison run with a supervised model
```bash
python3 main/train.py \
	--name Supervised_Exp_0 \                       # specify experiment name
	--data_root './data/' \                         # specify where you put the CORe50 dataset
	--contrast 'nocontrast' \                       # choose 'nocontrast' for supervised experiments
	--main_loss 'supervised' \                      # supervised loss
	--test_every 10 \                               # test every 10 epochs
	--train_split train_alt_0 \                     # choose the splits for cross-validation (k in range(5))
	--test_split test_alt_0 \
	--val_split val_alt_0 \

```

#### Run the experiments of the paper

There are several slurm sbatch scripts to run exactly the runs that are presented in the ICDL contribution. These scripts can be found under ./main.
Execute one of the scripts script to start a batch job using the slurm job manager that includes all runs presented in a specific Figure or Table,
e.g. fig3_foobar.sh includes all runs referenced in fig 3.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details