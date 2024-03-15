import os
import sys
import lmdb
import pickle
import os.path as osp
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from learning.utils.datasets import SimpleTimeContrastiveDataset

def raw_reader(path):
    with open(path, 'rb') as f:
        bin_data = f.read()
    return bin_data

def dump_pickle(obj):
    """
    Serialize an object.
    
    Returns :
        The pickled representation of the object obj as a bytes object
    """
    return pickle.dumps(obj)

def folder2lmdb(dpath, name="train_images", write_frequency=5000, num_workers=0):
    directory = osp.expanduser(osp.join(dpath, name))
    print("Loading dataset from %s" % directory)
    dataset = ImageFolder(directory, loader=raw_reader)
    
    dataset = SimpleTimeContrastiveDataset(directory, contrastive=False, loader=raw_reader)
    dataset.label_by = 'all'
    data_loader = DataLoader(dataset, num_workers=num_workers)

    lmdb_path = osp.join(dpath, "%s.lmdb" % name)
    isdir = os.path.isdir(lmdb_path)

    print("Generating LMDB to %s" % lmdb_path)
    map_size = 4e9 # this should be adjusted based on OS/db size #30737418240
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=map_size, readonly=False,
                   meminit=False, map_async=True)
    
    print(len(dataset), len(data_loader))
    txn = db.begin(write=True)
    for idx, (data, labels) in enumerate(data_loader):
        # print(type(data), data)
        image = data
        for i in range(len(labels)):
            labels[i] = np.squeeze(labels[i].numpy())
            #print(labels[i].shape)
        
        #label = label.numpy()
        #txn.put(u'{}'.format(idx).encode('ascii'), dump_pickle((image, label)))
        txn.put(u'{}'.format(idx).encode('ascii'), dump_pickle((image, labels[0], labels[1], labels[2], labels[3])))

        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dump_pickle(keys))
        txn.put(b'__len__', dump_pickle(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str)
    parser.add_argument('-s', '--split', type=str, default="train")
    parser.add_argument('--out', type=str, default=".")
    parser.add_argument('-p', '--procs', type=int, default=0)

    args = parser.parse_args()

    folder2lmdb(args.folder, num_workers=args.procs, name=args.split)
