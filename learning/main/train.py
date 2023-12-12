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

# configuration module
# -----

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
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
from utils.evaluation import fuse_representations, lls_fit, lls_eval, supervised_eval, knn_eval, wcss_bcss, get_pacmap, train_linear_classifier, evaluate, log_to_tensorboard
from utils.visualization import ConfusionMatrix
#from PIL import Image, ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True


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
}




# custom functions
# -----

def main():
    
    print(f"[INFO:] Using {device} backend.")
    
    run_name = f'{datetime.datetime.now().strftime("%d-%m-%y_%H:%M")}_{args.name}_seed_{args.seed}_{args.dataset}_aug_{args.contrast}_{args.main_loss}_reg_{args.reg_loss}'
    
    path_to_experiment = os.path.join(args.log_dir, run_name)
    # make directory
    os.makedirs(path_to_experiment, exist_ok=True)
    # write config to a json file
    save_args(path_to_experiment, args.__dict__)

    # tensorboard writer
    writer = SummaryWriter(log_dir=path_to_experiment)
    # global variable we keep track of
    loss_meter = AverageMeter()
    # confusion matrix visualization object
    conf_mat = ConfusionMatrix(n_cls=data_properties_dict[args.dataset].n_classes)
    # pacmap visualization object
    # TODO: write a pacmap class
    
    # get the dataloaders for the dataset
    dl_train, dl_train_eval, dl_test = get_dataloaders(
        args, data_properties_dict)
    
    # get the network
    network = get_network(args, data_properties_dict).to(device)
    
    # get the optimizer
    optimizer = get_optimizer(network, args)
    
    # get the scheduler
    scheduler = get_scheduler(optimizer, args)
        
    # initial evaluation (before training, epoch 0)
    results = evaluate(
        dataloader_test=dl_test,
        dataloader_train=dl_train,
        dataloader_train_eval=dl_train_eval,
        data_properties_dict=data_properties_dict,
        model=network,
        args=args,
        epoch=-1,
        writer=writer,
        confusion_matrix=conf_mat,
        device=device)
    
    
        
    # main training loop
    # -----
    
    epoch_loop = tqdm(range(args.n_epochs), ncols=80)
    for epoch in epoch_loop:
        loss_meter.reset()
        for (x, x_pair), y in dl_train:
            x, y = torch.cat([x, x_pair], 0).to(device), y.to(device)
            representation, projection = network(x)
            projection, pair = projection.split(projection.shape[0]//2)
            
            loss = get_loss(projection, pair, y, args)
            # set gradients to zero before next iterations
            optimizer.zero_grad(set_to_none=True)
            # backpropagate the loss
            loss.backward()
            optimizer.step()
            epoch_loop.set_description(f'Loss: {loss.item():>8.4f}')
            loss_meter.update(loss.item())
        
        # always write down the loss
        writer.add_scalar('accloss/train/loss',
                          loss_meter.avg, epoch + 1)

        # update learning rate
        update_lr(scheduler, epoch, args)

        # save model at the end of training and at given interval
        if (epoch + 1) % args.save_every == 0 or (epoch+1) == args.n_epochs:
            save_model(network, writer, epoch + 1, model_name='backbone')
        
        # test model at the end of training and at given interval
        if (epoch + 1) % args.test_every == 0 or (epoch + 1) == args.n_epochs:
            writer.add_scalar('analytics/learningrate',
                scheduler.get_last_lr()[0], epoch + 1)
            
            results = evaluate(
                    dataloader_test=dl_test,
                    dataloader_train=dl_train,
                    dataloader_train_eval=dl_train_eval,
                    data_properties_dict=data_properties_dict,
                    model=network,
                    args=args,
                    epoch=epoch,
                    writer=writer,
                    confusion_matrix=conf_mat,
                    device=device)
    
# ----------------
# main program
# ----------------

if __name__ == '__main__':
    main()

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
