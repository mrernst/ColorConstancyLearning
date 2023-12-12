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
import time


from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
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
        self.classes = [str(c) for c in range(0,self.n_classes)]
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
            light_colors = np.loadtxt(self.root + self.subdirectory +"/labels/" + f"light_colors_{int(object)}.txt", dtype=str)
            
            light_colors = np.stack([(np.stack([light_code_to_colorrgb(c) for c in light_colors[r]]).astype(int)) for r in range(len(light_colors))])
            
            light_powers = np.loadtxt(self.root+self.subdirectory +"/labels/" + f"light_powers_{int(object)}.txt")
            
            
            list_of_files = os.listdir(os.path.join(d, object))
            list_of_files.sort(key=lambda x: int((x.split('_')[-1].split('.')[0])))
            for timestep, path in enumerate(list_of_files):
                full_path = os.path.join(d, object, path)
                img_id = int((full_path.split('/')[-1].split('.')[0]))
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




class SimpleTimeContrastiveDataset(datasets.ImageFolder):
    # TODO: Rewrite this class from the ground up. I want this to be way more pythonic and use a predefined dict instead of anything else that always gives out the same pair instead of sampling for simplicity and fastness sake. Inherit from ImageFolder directly
    def __init__(
        self,
        root,
        contrastive,
        extensions = None,
        transform = None,
        target_transform = None,
        is_valid_file = None,
    ) -> None:
    
        super().__init__(root, transform=transform, target_transform=target_transform, is_valid_file=is_valid_file)
        # samples needs to be sorted first and the rotation needs to happen per object
        self.samples_shifted_plus_one = self.samples[1:] + self.samples[0:1]
        self.samples_shifted_minus_one = self.samples[-1:] + self.samples[:-1]
        # on a per object basis one could also make a definite list  l[1:] + l[-2:-1]
        self.contrastive = contrastive
        self.n_classes = len(self.classes)
        # load additional label information, i.e. lighting and light color
        self.load_additional_labels()
        self._label_by = 'object'

    
    def load_additional_labels(self):
        # have an ordered list
        list_of_objects = os.listdir(self.root)
        list_of_objects.sort(key=lambda x: int(x))
        
        light_colors_dict = {}
        light_powers_dict = {}
        
        for object in list_of_objects:
            
            # read in additional label data for lights and intensities
            # represent intensities as 8D vector
            # represent lights as RGB * light number 3*8=24 one-hot encoding
            light_colors = np.loadtxt(self.root.rsplit('/', 1)[0] +"/labels/" + f"light_colors_{int(object)}.txt", dtype=str)
            
            light_colors = np.stack([(np.stack([light_code_to_colorrgb(c) for c in light_colors[r]]).astype(int)) for r in range(len(light_colors))])
            
            light_powers = np.loadtxt(self.root.rsplit('/', 1)[0] +"/labels/" + f"light_powers_{int(object)}.txt")
            
            light_colors_dict[int(object)] = light_colors
            light_powers_dict[int(object)] = light_powers
            
        self.color_labels = []
        self.power_labels = []
        self.object_labels = []
        for full_path, label in self.samples:
            img_id = int((full_path.split('/')[-1].split('.')[0]))
            self.color_labels.append([light_colors_dict[label][img_id]])
            self.power_labels.append(light_powers_dict[label][img_id])
            self.object_labels.append(label)
        
        pass
    
    @property
    def label_by(self):
        return self._label_by
    
    @label_by.setter
    def label_by(self, l):
        assert l in ['object', 'color', 'power']        
        self._label_by = l
        if l =='color':
            self.n_classes = 18 # not real classes but output nodes
            self.classes = [f'{i}' for i in range(18)]
            # redo samples samples + 1 and samples - 1
            
            self.samples = [(a, b) for a, b in zip([tup[0] for tup in self.samples],self.color_labels)]
        elif l == 'power':
            self.n_classes = 6 # not real classes but output nodes
            self.classes = [f'{i}' for i in range(6)]
            self.samples = [(a, b) for a, b in zip([tup[0] for tup in self.samples],self.power_labels)]
        else:
            self.n_classes = 50
            self.classes = [f'{i}' for i in range(50)]
            self.samples = [(a, b) for a, b in zip([tup[0] for tup in self.samples],self.object_labels)]
        
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        

        if self.contrastive:
            # get contrast and check whether it is okay iff we sample contrasts
            if np.random.random() > 0.5:
                path2, target2 = self.samples_shifted_plus_one[index]
                if target2 != target:
                   path2, target2 = self.samples_shifted_minus_one[index]
            else:
                path2, target2 = self.samples_shifted_minus_one[index]
                if target2 != target:
                   path2, target2 = self.samples_shifted_plus_one[index]
            
            # path2, target2 = self.samples_shifted_plus_one[index]
            # if target2 != target:
            #    path2, target2 = self.samples_shifted_minus_one[index]
            sample = self.loader(path)
            augmentation = self.loader(path2)
            if self.transform is not None:
                sample = self.transform(sample)
                augmentation = self.transform(augmentation)
            if self.target_transform is not None:
                target = self.target_transform(target)
        
            output = ([sample, augmentation], target)
        else:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            
            output =  sample, target

        return output
    
    def __len__(self) -> int:
        return len(self.samples)

# ----------------
# main program
# ----------------

if __name__ == "__main__":
    
    
    
    dataset = SimpleTimeContrastiveDataset(
        root='../data/C3/train/',
        transform=transforms.ToTensor(),
        contrastive=True,
    )
    
    dataloader = DataLoader(dataset, batch_size=100, num_workers=0, shuffle=False)
    for ibatch, sample_batched in enumerate(dataloader):
        #print(ibatch)
        #print(sample_batched[0][0].shape)
    
        show_batch(sample_batched)
        if ibatch == 4:
            break
    
    
    
    start = time.time()
    for ibatch, sample_batched in enumerate(dataloader):
        pass
    end = time.time()
    print(f"SimpleTimeContrastive: {end - start}")
    # this is already 3x faster

    
    # C3 Dataset
    # -----
#     
#     dataset = C3Dataset(
#         root='../data',
#         split='train',
#         transform=transforms.ToTensor(),
#         contrastive=True,
#     )
#     
#     # original timeseries
#     dataloader = DataLoader(dataset, batch_size=100, num_workers=0, shuffle=False)
#     for ibatch, sample_batched in enumerate(dataloader):
#         #print(ibatch)
#         #print(sample_batched[0][0].shape)
# 
#         show_batch(sample_batched)
#         if ibatch == 4:
#             break
#     
#     start = time.time()
#     for ibatch, sample_batched in enumerate(dataloader):
#         pass
#     end = time.time()
#     print(f"TimeContrastive: {end - start}")
#     
#     
#     
#     
#     
#     # ---
#     
#     
#     
#     features = []
#     labels = []
#     for data_samples, data_labels in dataloader:   
#         features.append(data_samples[0])
#         labels.append(data_labels)
#     features = torch.cat(features, 0)
#     labels = torch.cat(labels, 0)
#     
#     features = features.reshape(features.shape[0], -1)
#     labels = labels.reshape(labels.shape[0], -1)
#  
#     
#     pacmap_plot = get_pacmap(
#     features, labels, 0, dataset.n_classes, dataset.classes)
#     #sys.exit()
#     
#     plt.show()
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
