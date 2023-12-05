#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import torch
import numpy as np
import os, sys
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

# utilities
# -----

# custom functions
# -----
def show_batch(sample_batched):
    """
    sample_batched: Tuple[torch.tensor, torch.tensor] -> None
    show_batch takes a contrastive sample sample_batched and plots
    an overview of the batch
    """

    grid_border_size = 2
    nrow = 10

    batch_1 = sample_batched[0][0][:, 0:, :, :]
    batch_2 = sample_batched[0][1][:, 0:, :, :]
    difference = np.abs(batch_1 - batch_2)

    titles = ["first contrast", "second contrast", "difference"]

    fig, axes = plt.subplots(1, 3, figsize=(2 * 6.4, 4.8))
    for (i, batch) in enumerate([batch_1, batch_2, difference]):
        ax = axes[i]
        grid = utils.make_grid(batch, nrow=nrow, padding=grid_border_size)
        ax.imshow(grid.numpy().transpose((1, 2, 0)))
        ax.set_title(titles[i])
        ax.axis("off")
    plt.show()


COLOR_DICT = {
        'B': ([0.,0.,0.], 'black'),
        'W': ([1.,1.,1.], 'white'),
        'R': ([1.,0,0], 'red'),
        'B': ([0,1.,0], 'blue'),
        'G': ([0,0,1.], 'green'),
        'M': ([1.,0,1.], 'magenta'),
        'C': ([0,1.,1.], 'cyan'),
        'Y': ([1.,1.,0], 'yellow'),
        'O': ([0,0,0], 'black')
    }
    
def light_code_to_colorrgb(code):
    return COLOR_DICT[code][0]
    
def light_code_to_colorname(code):
    return COLOR_DICT[code][1]


# ----------------
# custom classes
# ----------------

# custom CLTT dataset superclass (abstract)
# -----

class TimeContrastiveDataset(Dataset):
    """
    TimeContrastiveDataset is an abstract class implementing all methods necessary
    to sample data according to a time-contrastive approach. TimeContrastiveDataset itself
    should not be instantiated as a standalone class, but should be
    inherited from and abstract methods should be overwritten. If works witout sampling images to a buffer and is a simpler implementation.
    """  
    
    def __init__(self, root, split='train', transform=None,
        target_transform=None, contrastive=True):
        """
        __init__ initializes the CLTTDataset Class, it defines class-wide
        constants and builds the registry of files and the data buffer
        
        root:str path to the dataset directory
        split:str supports 'train', 'test', 'val'
        transform:torchvision.transform
        target_transform:torchvision.transform
        contrastive:bool contrastive dataset mode
        
        """
        super().__init__()
        
        # check if split is string and it is one of the valid options
        valid_splits = ['train', 'test', 'val']
        # add alternative split with potential k crossfolds
        valid_splits += [v+f'_alt_{k}' for v in valid_splits for k in range(5)]
        assert isinstance(split, str) and split in valid_splits, f'variable split has to be of type str and one of the following, {valid_splits}'
        
        self.split = split
        self.tau_plus = 1
        self.tau_minus = 1  # contrasts from the past avoid problems at the end of each object folder
                
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
    
        self.contrastive = contrastive
            
        self.get_dataset_properties()
        
        self.registry = self.build_registry(split)

        pass
    
    def __len__(self):
        """
        __len__ defines the length of the dataset and indirectly
        defines how many samples can be drawn from the dataset
        in one epoch
        """
        length = len(self.registry)
        return length
    
    def get_dataset_properties(self):
        """
        get_dataset_properties has to be defined for each dataset
        it stores number of objects, number of classes, a list of
        strings with labels
        """
        
        # basic properties (need to be there)
        self.n_objects = 3  # number of different objects >= n_classes
        self.n_classes = 3  # number of different classes
        self.labels = [
            "A",
            "B",
            "C",
            ]
        self.n_views_per_object = 10 # how many overall views of each object
        self.subdirectory = '/dataset_name/' # where is the dataset
        self.name = 'dataset name' # name of the dataset
        
        # custom properties (optional, dataset specific)
        # (anything you would want to have available in self)
        self.custom_property = ['one', 'two', 'three']
        
        raise Exception("Calling abstract method, please inherit \
        from the CLTTDataset class and reimplement this method") # pseudoclass 
        pass
    
        
    def __getitem__(self, idx):
        """
        __getitem__ is a method that defines how one sample of the
        dataset is drawn
        """
        if self.contrastive:
            image, label = self.get_single_item(idx)
            augmentation, _ = self.sample_contrast(idx)
        
            if self.transform:
                image, augmentation = self.transform(
                    image), self.transform(augmentation)
        
            if self.target_transform:
                label = self.target_transform(label)
        
            output = ([image, augmentation], label)
        else:
            image, label = self.get_single_item(idx)
        
            if self.transform:
                image = self.transform(image)
        
            if self.target_transform:
                label = self.target_transform(label)
        
            output = image, label
        
        return output
    
    def sample_contrast(self, chosen_index):
        """
        given index chosen_index, sample a corresponding contrast close in time
        """
        chosen_time = self.registry.iloc[chosen_index]["time_idx"]
        
        possible_indices = self.registry[
            (self.registry["time_idx"].between(chosen_time - self.tau_minus, chosen_time + self.tau_plus)) & (
                    self.registry["time_idx"] != chosen_time) & (self.registry['object_nr'] == self.registry[self.registry['time_idx']== chosen_time]['object_nr'].values[0])
                ].index
        
        chosen_index = np.random.choice(possible_indices)
        return self.get_single_item(chosen_index)
    
    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")
    
    def get_single_item(self, idx):
        """
        Given a single index idx return image and label
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # elif isinstance(idx, pd.core.indexes.numeric.Int64Index):
        #     idx = idx[0]
        
        path_to_file = self.registry.loc[idx, "path_to_file"]
        if isinstance(path_to_file, pd.core.series.Series):
            path_to_file = path_to_file.item()
        
        image = self.pil_loader(path_to_file)
        obj_info = self.registry.iloc[idx, 1:].to_dict()
        
        label = self.registry.loc[idx, "label"]
        return image, label
    
    def build_registry(self, split):
        """
        build a registry of all image files of the dataset
        """
        pass

        path_list = []
        object_list = []
        label_list = []
        time_list = []
        
        if split == 'train':
            d = self.root + self.subdirectory + 'train/'
            assert os.path.isdir(d), 'Train directory does not exist'
        elif split == 'test':
            d = self.root + self.subdirectory + 'test/'
            assert os.path.isdir(d), 'Test directory does not exist'
        else:
            d = self.root + self.subdirectory + 'val/'
            if not(os.path.isdir(d)):
                print('[INFO] Validation directory does not exist, using testset instead')
                d = self.root + self.subdirectory + 'test/'
                
        
        # have an ordered list
        list_of_objects = os.listdir(d)
        list_of_objects.sort(key=lambda x: int(x))
                
        for i, object in enumerate(list_of_objects):
            list_of_files = os.listdir(os.path.join(d, object))
            list_of_files.sort(key=lambda x: int((x.split('_')[-1].split('.')[0])))
            for timestep, path in enumerate(list_of_files):
                full_path = os.path.join(d, object, path)
                if os.path.isfile(full_path):
                    path_list.append(full_path)
                    object_list.append(object)
                    label_list.append(object)
                    time_list.append(timestep+i*self.n_views_per_object)
            
            # read in additional label data for lights and intensities
            # represent intensities as 8D vector
            # represent lights as 8*8=64 one-hot encoding
            np.load_txt()
                
        tempdict = {'path_to_file': path_list, 'label': label_list, 'object_nr': object_list, 'time_idx': time_list}
        
        
        dataframe = pd.DataFrame(tempdict)
        dataframe.sort_values(by=['object_nr', 'time_idx'], inplace=True)
        dataframe.reset_index(drop=True, inplace=True)
        
        
        return dataframe


class C3Dataset(TimeContrastiveDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass
    
    def get_dataset_properties(self):
        """
        get_dataset_properties has to be defined for each dataset
        it stores number of objects, number of classes, a list of
        strings with labels
        """
        
        # basic properties (need to be there)
        self.n_objects = 50  # number of different objects >= n_classes
        self.n_classes = 50  # number of different classes
        self.labels = [str(c) for c in range(0,self.n_classes)]
        self.n_views_per_object = 300 # how many overall views of each object #TODO: this should be 1000 or more
        self.subdirectory = '/C3/' # where is the dataset
        self.name = 'ColorConstancyCubes' # name of the dataset
        pass
    
    def build_registry(self, split):
        """
        build a registry of all image files of the dataset
        """
        pass
        
        path_list = []
        object_list = []
        label_list = []
        time_list = []
        
        light_power_list = []
        light_color_list = []
        
        if split == 'train':
            d = self.root + self.subdirectory + 'train/'
            assert os.path.isdir(d), 'Train directory does not exist'
        elif split == 'test':
            d = self.root + self.subdirectory + 'test/'
            assert os.path.isdir(d), 'Test directory does not exist'
        else:
            d = self.root + self.subdirectory + 'val/'
            if not(os.path.isdir(d)):
                print('[INFO] Validation directory does not exist, using testset instead')
                d = self.root + self.subdirectory + 'test/'
                
        
        # have an ordered list
        list_of_objects = os.listdir(d)
        list_of_objects.sort(key=lambda x: int(x))
                
        for i, object in enumerate(list_of_objects):
            
            # read in additional label data for lights and intensities
            # represent intensities as 8D vector
            # represent lights as RGB * light number 3*8=24 one-hot encoding
            light_colors = np.loadtxt(self.root + self.subdirectory +"/labels/" + f"light_colors_{object}.txt", dtype=str)
            
            light_colors = np.stack([(np.stack([light_code_to_colorrgb(c) for c in light_colors[r]]).astype(int)) for r in range(len(light_colors))])
            
            light_powers = np.loadtxt(self.root+self.subdirectory +"/labels/" + f"light_powers_{object}.txt")
            
            
            list_of_files = os.listdir(os.path.join(d, object))
            list_of_files.sort(key=lambda x: int((x.split('_')[-1].split('.')[0])))
            for timestep, path in enumerate(list_of_files):
                full_path = os.path.join(d, object, path)
                img_id = int((full_path.split('_')[-1].split('.')[0]))
                if os.path.isfile(full_path):
                    path_list.append(full_path)
                    object_list.append(i)
                    label_list.append(i)
                    time_list.append(img_id+i*self.n_views_per_object)
                    light_color_list.append([light_colors[img_id]])
                    light_power_list.append(light_powers[img_id])
                

                
        tempdict = {'path_to_file': path_list, 'label': label_list, 'object_nr': object_list, 'time_idx': time_list,
            'light_color': light_color_list, 'light_power':light_power_list}
        
        
        dataframe = pd.DataFrame(tempdict)
        dataframe.sort_values(by=['object_nr', 'time_idx'], inplace=True)
        dataframe.reset_index(drop=True, inplace=True)
        
        
        return dataframe
    

# TODO: add a C3 neutral dataset that uses a simulated buffer again, just build the registry by copying all entries a number of times so we can compare with the regular dataset


# ----------------
# main program
# ----------------

if __name__ == "__main__":
    
    # CORE50 Dataset
    # -----
    
    dataset = C3Dataset(
        root='../data',
        split='train',
        transform=transforms.ToTensor(),
        contrastive=True,
    )
    
    # original timeseries
    dataloader = DataLoader(dataset, batch_size=100, num_workers=0, shuffle=False)
    for ibatch, sample_batched in enumerate(dataloader):
        print(ibatch)
        print(sample_batched[1])

        show_batch(sample_batched)
        if ibatch == 4:
            break
    

    sys.exit()

#  _____________________________________________________________________________

# Stick to 80 characters per line
# Use PEP8 Style
# Comment your code

# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
