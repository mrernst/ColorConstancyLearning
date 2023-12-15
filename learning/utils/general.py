#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import os
import math
import torch

# configuration module
# -----
import json

# custom classes
# -----

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AnalogueLogger(object):
    pass
    
    
class TwoContrastTransform:
    """
    Create two contrasts of the same image using the given
    torchvision transform
    """
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


# custom functions
# -----

def save_model(model, writer, epoch, model_name=''):
    """ Save model parameters.
    Args:
        model:
            network to be saved
        writer:
            tensorboard summary writer to get the log directory
        epoch:
            current epoch represented by an integer
    """
    log_dir = writer.get_logdir()
    path = os.path.join(log_dir, 'models')
    if not os.path.exists(path):
        os.mkdir(path)
    torch.save(model.state_dict(), os.path.join(path, f'{model_name}_epoch_{epoch}.pt'))


def load_model(model, path, device):
    """ Load model parameters.
    
    Args:
        model:
            network to load the parameters.
        path:
            path to the saved model state dict represented by a
            string object.
        device:
            the device you want to load the model onto,
            represented by a string object
    """
    
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    
    pass


def save_args(path, args):
    """Save command line arguments to json file.
    
    Args:
        path: Filepath to the desired output directory represented by a string.
        args: parser.args object.
    """
    filename = 'args.json'
    # Save args
    l = []
    for a in args:
      try:
        l.append(args[a].items())
      except:
        l.append((a,args[a]))
    with open(os.path.join(path, filename), 'w') as f:
        json.dump(l, f, indent=2)
    
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