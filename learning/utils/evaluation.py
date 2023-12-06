#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import torch
from torch.linalg import lstsq
import torch.nn.functional as F
import pacmap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
from tqdm import tqdm

from utils.visualization import ConfusionMatrix
from utils.networks import LinearClassifier, SplitOutput
# custom functions
# -----
@torch.no_grad()
def get_representations(model, data_loader, device='cpu'):
    """
    Get all representations of the dataset given the network and the data loader
    params:
        model: the network to be used (torch.nn.Module)
        data_loader: data loader of the dataset (DataLoader)
    return:
        representations: representations output by the network (Tensor)
        labels: labels of the original data (LongTensor)
    """
    model.eval()
    features = []
    labels = []
    for data_samples, data_labels in data_loader:
        #print(data_labels)
        features.append(model(data_samples.to(device))[0])
        labels.append(data_labels.to(device))
    features = torch.cat(features, 0)
    labels = torch.cat(labels, 0)
    return features, labels


@torch.no_grad()
def lls_fit(train_features, train_labels, n_classes):
    """
        Fit a linear least square model
        params:
            train_features: the representations to be trained on (Tensor)
            train_labels: labels of the original data (LongTensor)
            n_classes: int, number of classes
        return:
            ls: the trained lstsq model (torch.linalg) 
    """
    ls = lstsq(train_features, F.one_hot(train_labels, n_classes).type(torch.float32))
    
    return ls

@torch.no_grad()
def lls_eval(trained_lstsq_model, eval_features, eval_labels):
    """
    Evaluate a trained linear least square model
    params:
        trained_lstsq_model: the trained lstsq model (torch.linalg)
        eval_features: the representations to be evaluated on (Tensor)
        eval_labels: labels of the data (LongTensor)
    return:
        acc: the LLS accuracy (float)
    """
    prediction = (eval_features @ trained_lstsq_model.solution)
    acc = (prediction.argmax(dim=-1) == eval_labels).sum() / len(eval_features)
    return prediction, acc



def train_linear_classifier(train_dataloader, test_dataloader, val_dataloader, input_features, num_classes, backbone, epochs=200, global_epoch=0, test_every=1, writer=None, device='cpu'):
    print(f'[INFO:] Starting linear evaluation with Neural Network at epoch {global_epoch}')
    # tune the backbone
    backbone.eval()
    #define model loss and optimizer
    classifier = LinearClassifier(input_features, num_classes).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    
    model = torch.nn.Sequential(
      backbone,
      SplitOutput(0),
      classifier
    ).to(device)
    
    
    training_loop = tqdm(range(epochs), ncols=80)
    
    for t in training_loop:
        training_loss, training_acc = train(train_dataloader, model, loss_fn, optimizer, device=device)
        if t%test_every==0:
            testing_loss, testing_acc = test(test_dataloader, model, loss_fn, device=device)
            training_loop.set_description(f'Loss: {testing_loss:>8.4f}')
            validation_loss, validation_acc = test(val_dataloader, model, loss_fn, device=device)

            
            if writer:
                # if conditions to make logging more
                # readable if we decide to only log the last
                # timestep -> learning curve
                description_string = 'linearNN/'
                if test_every == epochs:
                    logged_epoch = global_epoch
                else:
                    logged_epoch = t
                    description_string += f"/e{global_epoch}"

                # log data to tensorboard writer
                writer.add_scalar(description_string + f'/accloss/train/loss', training_loss, logged_epoch + 1)
                writer.add_scalar(description_string + f'/accloss/test/loss', testing_loss, logged_epoch + 1)
                writer.add_scalar(description_string + f'/accloss/val/loss', testing_loss, logged_epoch + 1)
                writer.add_scalar(description_string + f'/accloss/train/accuracy', training_acc, logged_epoch + 1)
                writer.add_scalar(description_string + f'/accloss/test/accuracy', testing_acc, logged_epoch + 1)
                writer.add_scalar(description_string + f'/accloss/val/accuracy', testing_acc, logged_epoch + 1)
            
                
    
    return testing_loss, testing_acc


def train(dataloader, model, loss_fn, optimizer, device='cpu'):
    size = len(dataloader.dataset)
    model.train()
    num_batches = len(dataloader)
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # print(batch)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # if batch % 10 == 0:
        #     loss, current = loss.item(), (batch + 1) * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    train_loss /= num_batches
    correct /= size
        # return losses and accuracies
    return train_loss, correct
        
        
        
def test(dataloader, model, loss_fn, cm=None, device='cpu'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            if cm:
                cm.update(pred.cpu(), y.cpu())
    test_loss /= num_batches
    correct /= size
    # print(f"[INFO:] Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return test_loss, correct



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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the
    specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
    
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
    
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1. / batch_size))
        return res

@torch.no_grad()
def supervised_eval(model, dataloader, criterion, no_classes, device='cpu'):
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    confusion_matrix = ConfusionMatrix(n_cls=no_classes)
    
    for idx, (images, labels) in enumerate(dataloader):
        images = images.float()
        
        images = images.to(device)
        labels = labels.to(device)
        bsz = labels.shape[0]
        
        # forward
        _, output = model(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        
        top1.update(acc1[0], bsz)
        
        # update confusion matrix
        confusion_matrix.update(output.cpu(), labels.cpu())
        
    print('Test: [{0}/{1}]\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
              idx, len(dataloader),
              loss=losses, top1=top1))
        
    return top1.avg, losses.avg, confusion_matrix
    

@torch.no_grad()
def wcss_bcss(representations, labels, n_classes):
    """
        Calculate the within-class and between-class average distance ratio
        params:
            representations: the representations to be evaluated (Tensor)
            labels: labels of the original data (LongTensor)
        return:
            wb: the within-class and between-class average distance ratio (float)
    """
    # edgecase: there might be less on one class than of another
    # -----
    # representations = torch.stack([representations[labels == i] for i in range(n_classes)])
    # centroids = representations.mean(1, keepdim=True)
    # wcss = (representations - centroids).norm(dim=-1).mean()
    # bcss = F.pdist(centroids.squeeze()).mean()
    # wb = wcss / bcss
    representations = [representations[labels == i] for i in range(n_classes)]
    centroids = torch.stack([r.mean(0, keepdim=True) for r in representations])
    wcss = [(r - centroids[i]).norm(dim=-1) for i,r in enumerate(representations)]
    wcss = torch.cat(wcss).mean()
    bcss = F.pdist(centroids.squeeze()).mean()
    wb = wcss / bcss
    return wb



#https://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Unsupervised_Feature_Learning_CVPR_2018_paper.pdf
def knn_eval(train_features, train_labels, test_features, test_labels, n_classes, sim_function, size_batch=512):
    k=20
    i=0
    correct=0
    probs_list = []
    expanded_train_label = train_labels.view(1,-1).expand(size_batch,-1)
    retrieval_one_hot = torch.zeros(size_batch*k, n_classes,device=train_features.device)
    while i < test_features.shape[0]:
        endi = min(i+size_batch,test_features.shape[0])
        tf = test_features[i:endi]
        distance_matrix = sim_function(tf, train_features)
        valk, indk = torch.topk(distance_matrix, k, dim=1)
        # valk is batchsize x k
        retrieval =  torch.gather(expanded_train_label, 1, indk)
        #retrieval_one_hot = torch.zeros(K, C)
        #List of labels [1, 4, 9...]
        if retrieval.shape[0] < size_batch:
            retrieval_one_hot = torch.zeros(tf.shape[0]*k, n_classes, device=train_features.device)
        retrieval_one_hot[:,:]=-10000
        scattered_retrieval = retrieval_one_hot.scatter_(1, retrieval.view(-1,1) , 1)
        retrieval_onehot = scattered_retrieval.view(retrieval.shape[0],k, n_classes)
        sim_topk = retrieval_onehot*valk.unsqueeze(-1)
        probs = torch.sum(sim_topk, dim=1)
        probs_list.append(probs)
        prediction = torch.max(probs,dim=1).indices
        correct += (prediction == test_labels[i:endi]).sum(dim=0)
        i=endi

    return torch.cat(probs_list), correct/test_features.shape[0]


@torch.no_grad()
def get_pacmap(representations, labels, epoch, n_classes, class_labels):
    """
        Draw the PacMAP plot
        params:
            representations: the representations to be evaluated (Tensor)
            labels: labels of the original data (LongTensor)
            epoch: epoch (int)
        return:
            fig: the PacMAP plot (matplotlib.figure.Figure)
    """
    # sns.set()
    sns.set_style("ticks")
    sns.set_context('paper', font_scale=1.8, rc={'lines.linewidth': 2})
    # color_map = get_cmap('viridis')
    color_map = ListedColormap(sns.color_palette('colorblind', 50))
    #legend_patches = [Patch(color=color_map(i / n_classes), label=label) for i, label in enumerate(class_labels)]
    legend_patches = [Patch(color=color_map(i), label=label) for i, label in enumerate(class_labels)]
    # save the visualization result
    embedding = pacmap.PaCMAP(n_components=2)
    X_transformed = embedding.fit_transform(representations.cpu().numpy(), init="pca")
    fig, ax = plt.subplots(1, 1, figsize=(7.7,4.8))
    labels = labels.cpu().numpy()
    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=color_map(labels), s=0.6)
    ax.set_title('Pacmap Plot')
    plt.xticks([]), plt.yticks([])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    ax.legend(loc='upper left', bbox_to_anchor=(1., 1.), handles=legend_patches, fontsize=13.8)
    # ax.text(-0.15, 1.05, 'C', transform=ax.transAxes, size=30, weight='medium')
    plt.xlabel(f'Epoch: {epoch}')
    return fig


def cosine_similarity(p_vec, q_vec):
    """
    cosine_similarity takes two numpy arrays of the same shape and returns
    a float representing the cosine similarity between two vectors
    """
    p_vec, q_vec = p_vec.flatten(), q_vec.flatten()
    return np.dot(p_vec, q_vec) / (np.linalg.norm(p_vec) * np.linalg.norm(q_vec))


@torch.no_grad()
def get_neighbor_similarity(representations, labels, epoch, sim_func=cosine_similarity):
    """
        Draw a similarity plot
        params:
            representations: the representations to be evaluated (Tensor)
            labels: labels of the original data (LongTensor)
            epoch: epoch (int)
            sim_func: similarity function with two parameters
        return:
            fig: similarity plot (matplotlib.figure.Figure)
    """

    unique_labels = torch.unique(labels)

    if len(labels) != len(unique_labels):
        # calculate the mean over representations
        rep_centroid = torch.zeros([len(unique_labels), representations.shape[-1]])
        for i in range(len(unique_labels)):
            rep_centroid[i] = representations[torch.where(labels == i)[0]].mean(0)

        list_of_indices = np.arange(len(unique_labels))
        labels = list_of_indices
        representations = rep_centroid
        n_samples_per_object = 1

    else:
        list_of_indices = np.arange(len(labels))
        n_samples_per_object = 1

    distances = np.zeros([len(unique_labels), len(unique_labels)])

    # Fill a distance matrix that relates every representation of the batch
    for i in list_of_indices:
        for j in list_of_indices:
            distances[labels[i], labels[j]] += sim_func(representations[i].cpu(), representations[j].cpu())
            # distances[labels[i], labels[j]] += 1

    distances /= n_samples_per_object ** 2  # get the mean distances between representations

    # get some basic statistics
    # print('[INFO:] distance', distances.max(), distances.min(), distances.std())

    # duplicate the matrix such that you don't get to the edges when
    # gathering distances
    distances = np.hstack([distances, distances, distances])
    # plt.matshow(distances)
    # plt.show()

    # how many neighbors do you want to show (n_neighbors = n_classes for sanity check, you would have to see a global symmetry)
    n_neighbors = len(unique_labels)
    topk_dist_plus = np.zeros([len(labels), n_neighbors])
    topk_dist_minus = np.zeros([len(labels), n_neighbors])

    for k in range(n_neighbors):
        for i in range(len(unique_labels)):
            topk_dist_plus[i, k] += distances[i, i + len(unique_labels) + k]
            topk_dist_minus[i, k] += distances[i, i + len(unique_labels) - k]

    topk_dist = np.vstack([topk_dist_plus, topk_dist_minus])

    fig, ax = plt.subplots()
    ax.errorbar(np.arange(0, n_neighbors), topk_dist.mean(0), marker='.', markersize=10, xerr=None,
                yerr=topk_dist.std(0))
    ax.set_title('representation similarity')
    ax.set_xlabel('nth neighbour')
    ax.set_ylabel('cosine similarity')
    ax.set_ylim(-1.1, 1.1)
    ax.hlines(topk_dist.mean(0)[n_neighbors // 2:].mean(), -100, 100, color='gray', linestyle='--')
    ax.set_xlim(-2, n_neighbors + 2)

    return fig

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
