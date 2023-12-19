#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import torch
from torch.linalg import lstsq as torch_lstsq
from scipy.linalg import lstsq as scipy_lstsq
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pacmap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap, hsv_to_rgb

import seaborn as sns
import numpy as np
from tqdm import tqdm

from utils.visualization import ConfusionMatrix
from utils.networks import LinearClassifier
from utils.general import DotDict, AverageMeter, save_model
from utils.configurator import get_dataloaders


# custom functions
# -----


@torch.no_grad()
def fuse_representations(model, dataloader, device='cpu'):
    """
    Get all representations of the dataset given the network and the data loader
    params:
        model: the network to be used (torch.nn.Module)
        dataloader: data loader of the dataset (DataLoader)
        device: the computing device (str)
    return:
        representations: representations output by the network (Tensor)
        labels: labels of the original data (LongTensor)
    """
    
    model.eval()
    features = []
    labels = []
    
    for X, y in dataloader:      
        features.append(model(X.to(device))[0].detach().cpu())
        labels.append(y.cpu())
    features = torch.cat(features, 0)
    labels = torch.cat(labels, 0)
    return features, labels


@torch.no_grad()
def lls_fit(train_features, train_labels, num_classes, label_type='one_hot', scipy=False):
    """
        Fit a linear least square model
        params:
            train_features: the representations to be trained on (Tensor)
            train_labels: labels of the original data (LongTensor)
            num_classes: int, number of classes
        return:
            ls: the trained lstsq model (torch.linalg) 
    """
    
    if label_type == 'one_hot':
        labels = F.one_hot(train_labels.cpu(), num_classes).type(torch.float32)
    else:
        labels = train_labels.cpu().type(torch.float32)
    
    if scipy:
        # this scipy fallback is there because on the Cluster pytorch 2 only works in limited ways
        p, res, rnk, s = scipy_lstsq(train_features.cpu(), labels)
        # package in a dot_dict so that you can use it as if it was the torch version
        ls = DotDict({'solution':p, 'residuals':res, 'rank':rnk, 'singular_values':s})

    else:
        ls = torch_lstsq(train_features, labels)
    
    
    return ls

@torch.no_grad()
def lls_eval(trained_lstsq_model, eval_features, eval_labels, label_type='one_hot'):
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
    
    if label_type == 'one_hot':
        acc = (prediction.argmax(dim=-1) == eval_labels).sum() / len(eval_features)
    elif label_type == 'n_hot':
        # TODO: this is not general and only works
        # for the specific encoding of the dataset
        eval_labels = torch.reshape(eval_labels,(-1,8,3))
        prediction = torch.reshape(prediction,(-1,8,3))
        acc = ((((F.sigmoid(prediction)>0.5) == eval_labels).sum(2) == 3).sum(1)/8).sum() / prediction.shape[0]
        #acc = torch.sum(torch.eq(torch.sum(F.sigmoid(prediction>0.5) == eval_labels,2),3))/ prediction.shape[0]

    else:
        acc = None
    return prediction, acc


def train_linear_regressor(train_dataloader, test_dataloader, input_features, output_features, model, loss_fn=torch.nn.L1Loss(), learning_rate=1e-3, epochs=200, timestep=0, test_every=1, label_type='continous', writer=None, description="reg", device='cpu'):
    
    assert label_type in ['continous'], f"label_type {label_type} not allowed."
    #TODO: finish this function
    
    print(f'\n[INFO:] Training linear neural network regressor at epoch {timestep + 1}')
    regressor = LinearClassifier(input_features, output_features).to(device)
    optimizer = torch.optim.Adam(regressor.parameters(), lr=learning_rate)
    
    training_loop = tqdm(range(epochs), ncols=80)
    
    for t in training_loop:
        training_loss, training_acc = train(train_dataloader, model, regressor, loss_fn, optimizer, label_type, device)
        if (((t+1)%test_every==0) or (t==0) or (t+1==epochs)): # add results at the first and the last timestep in any case
            if (t+1 == epochs):
                # if we are in the last epoch evaluate with a confusion matrix
                testing_loss, testing_acc = test(test_dataloader, model, regressor, loss_fn, label_type, device=device)
                training_loop.set_description(f'Loss: {testing_loss:>8.4f}')
            else:
                testing_loss, testing_acc = test(test_dataloader, model, regressor, loss_fn, label_type, device=device)
                training_loop.set_description(f'Loss: {testing_loss:>8.4f}')
    
                
            if writer:
                description_string = f"linear/{description}/e{timestep}"
    
                # log data to tensorboard writer
                writer.add_scalar(description_string + f'/accloss/train/loss', training_loss, t + 1)
                writer.add_scalar(description_string + f'/accloss/test/loss', testing_loss, t + 1)
    
    # save trained regressor
    if writer:
        save_model(regressor, writer, timestep, model_name='regressor')
    # reset backbone to training
    model.train()
    # TODO: add an evaluation metric that you could return instead of accuracy
    return training_loss, None, testing_loss, None


def train_linear_classifier(train_dataloader, test_dataloader, input_features, output_features, model, loss_fn=torch.nn.CrossEntropyLoss(), learning_rate=1e-3, epochs=200, timestep=0, test_every=1, label_type='one_hot', confusion_matrix=None, writer=None, description="class", device='cpu'):
    """
    """
    assert label_type in ['one_hot', 'n_hot'], f"label_type {label_type} not allowed."
    print(f'\n[INFO:] Training linear neural network classifier at epoch {timestep + 1}')
    #define model loss and optimizer
    classifier = LinearClassifier(input_features, output_features).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    
    
    training_loop = tqdm(range(epochs), ncols=80)
    
    for t in training_loop:
        training_loss, training_acc = train(train_dataloader, model, classifier, loss_fn, optimizer, label_type, device)
        if (((t+1)%test_every==0) or (t==0) or (t+1==epochs)): # add results at the first and the last timestep in any case
            if (t+1 == epochs):
                # if we are in the last epoch evaluate with a confusion matrix
                testing_loss, testing_acc = test(test_dataloader, model, classifier, loss_fn, label_type, confusion_matrix, device)
                training_loop.set_description(f'Loss: {testing_loss:>8.4f}')
                
                if (writer and confusion_matrix):
                    confusion_matrix.to_tensorboard(
                        writer, [f"{i:04}" for i in range(output_features)], timestep, label='test/class/nn_cm',)
                    confusion_matrix.reset()
            else:
                testing_loss, testing_acc = test(test_dataloader, model, classifier, loss_fn, label_type, device=device)
                training_loop.set_description(f'Loss: {testing_loss:>8.4f}')
    
                
            if writer:
                description_string = f"linear/{description}/e{timestep}"

                # log data to tensorboard writer
                writer.add_scalar(description_string + f'/accloss/train/loss', training_loss, t + 1)
                writer.add_scalar(description_string + f'/accloss/test/loss', testing_loss, t + 1)
                writer.add_scalar(description_string + f'/accloss/train/accuracy', training_acc, t + 1)
                writer.add_scalar(description_string + f'/accloss/test/accuracy', testing_acc, t + 1)
    
    # save trained classifier
    if writer:
        save_model(classifier, writer, timestep, model_name='classifier')
    # reset backbone to training
    model.train()
    return training_loss, training_acc, testing_loss, testing_acc


def train(dataloader, model, classifier, loss_fn, optimizer, label_type='one_hot', device='cpu'):
    size = len(dataloader.dataset)
    model.eval()
    classifier.train()
    num_batches = len(dataloader)
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        # print(batch)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        with torch.no_grad():
            features = model.encoder(X)
        pred = classifier(features.detach())
        
        loss = loss_fn(pred, y)
        
        train_loss += loss.item()
        if label_type == 'one_hot':
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        elif label_type == 'n_hot':
            pred = torch.nn.functional.sigmoid(pred)
            y, pred = torch.reshape(y,(-1,8,3)), torch.reshape(pred,(-1,8,3))
            correct += ((((pred>0.5) == y).sum(2) == 3).sum(1)/8).sum()
        else:
            correct = 0
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
        
        
@torch.no_grad()   
def test(dataloader, model, classifier, loss_fn, label_type='one_hot', confusion_matrix=None, device='cpu'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    classifier.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = classifier(model.encoder(X))
            test_loss += loss_fn(pred, y).item()
            if label_type == 'one_hot':
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            elif label_type == 'n_hot':
                pred = torch.nn.functional.sigmoid(pred)
                y, pred = torch.reshape(y,(-1,8,3)), torch.reshape(pred,(-1,8,3))
                correct += ((((pred>0.5) == y).sum(2) == 3).sum(1)/8).sum()
            else:
                correct = 0
            if confusion_matrix:
                confusion_matrix.update(pred.cpu(), y.cpu())
    test_loss /= num_batches
    correct /= size
    # print(f"[INFO:] Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return test_loss, correct



@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the
    specified values of k"""
    # TODO: This is too elaborate for what I'm mostly doing here, just get top1 accuracy
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
def supervised_eval(model, dataloader, criterion, no_classes, confusion_matrix=None, device='cpu'):
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
        
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
        if confusion_matrix:
            confusion_matrix.update(output.cpu(), labels.cpu())
        
    print('\nTest: [{0}/{1}]\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
              idx+1, len(dataloader),
              loss=losses, top1=top1))
    # TODO: change these misleading statistics
        
    return top1.avg, losses.avg
    

@torch.no_grad()
def wcss_bcss(representations, labels, num_classes):
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
    # representations = torch.stack([representations[labels == i] for i in range(num_classes)])
    # centroids = representations.mean(1, keepdim=True)
    # wcss = (representations - centroids).norm(dim=-1).mean()
    # bcss = F.pdist(centroids.squeeze()).mean()
    # wb = wcss / bcss
    representations = [representations[labels == i] for i in range(num_classes)]
    centroids = torch.stack([r.mean(0, keepdim=True) for r in representations])
    wcss = [(r - centroids[i]).norm(dim=-1) for i,r in enumerate(representations)]
    wcss = torch.cat(wcss).mean()
    bcss = F.pdist(centroids.squeeze()).mean()
    wb = wcss / bcss
    return wb



#https://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Unsupervised_Feature_Learning_CVPR_2018_paper.pdf
@torch.no_grad()
def knn_eval(train_features, train_labels, test_features, test_labels, num_classes, sim_function, size_batch=512):
    k=20
    i=0
    correct=0
    probs_list = []
    expanded_train_label = train_labels.view(1,-1).expand(size_batch,-1)
    retrieval_one_hot = torch.zeros(size_batch*k, num_classes, device=train_features.device)
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
            retrieval_one_hot = torch.zeros(tf.shape[0]*k, num_classes, device=train_features.device)
        retrieval_one_hot[:,:]=-10000
        scattered_retrieval = retrieval_one_hot.scatter_(1, retrieval.view(-1,1) , 1)
        retrieval_onehot = scattered_retrieval.view(retrieval.shape[0],k, num_classes)
        sim_topk = retrieval_onehot*valk.unsqueeze(-1)
        probs = torch.sum(sim_topk, dim=1)
        probs_list.append(probs)
        prediction = torch.max(probs, dim=1).indices
        correct += (prediction == test_labels[i:endi]).sum(dim=0)
        i=endi
    #print(probs_list, correct/test_features.shape[0])
    return torch.cat(probs_list), correct/test_features.shape[0]



@torch.no_grad()
def get_pacmap(representations, labels, epoch, num_classes, class_labels, color_map=None):
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
    if not(color_map):
        color_map = ListedColormap(sns.color_palette('colorblind', num_classes))
        #hsl_colors = [(i * (360 / num_classes), 50, 100) for i in range(num_classes)]
        #color_map = ListedColormap([hsv_to_rgb((hsl[0] / 360, hsl[1] / 100, hsl[2] / 100)) for hsl in hsl_colors])
    else:
        pass
    

    
    #legend_patches = [Patch(color=color_map(i / num_classes), label=label) for i, label in enumerate(class_labels)]
    legend_patches = [Patch(color=color_map(i), label=label) for i, label in enumerate(class_labels)]
    # save the visualization result
    embedding = pacmap.PaCMAP(n_components=2)
    X_transformed = embedding.fit_transform(representations.cpu().numpy(), init="pca")
    fig, ax = plt.subplots(1, 1, figsize=(14, 10), dpi=90, facecolor='w', edgecolor='k')
    
    labels = labels.cpu().numpy()
    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=color_map(labels), s=2.)
    ax.set_title('Pacmap Plot')
    plt.xticks([]), plt.yticks([])
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='upper left', bbox_to_anchor=(1., 1.), handles=legend_patches, fontsize=14, ncol=2)
    #legend(loc=2, prop={'size': 6})
    # ax.text(-0.15, 1.05, 'C', transform=ax.transAxes, size=30, weight='medium')
    plt.xlabel(f'Epoch: {epoch}')
    return fig




# high level functions
# -----

# these functions are basically just wrappers to code segments
# in train.py to make the training loop more readable

def evaluate(dataloader_test, dataloader_train_eval, dataloader_train, data_properties_dict, model, args, epoch, writer, confusion_matrix=None, device='cpu'):
    # return dict with with names and values
    eval_results_dict = DotDict({})
    # set model into evaluation mode
    model.eval()
    
    if args.main_loss == 'supervised':
        # supervised model evaluation
        test_acc, test_loss = supervised_eval(
            model, dataloader_test, F.cross_entropy, data_properties_dict[args.dataset].n_classes, confusion_matrix, device=device)
            
        writer.add_scalar(
            'accloss/test/class/accuracy', test_acc, epoch + 1)
        writer.add_scalar(
            'accloss/test/class/loss', test_loss, epoch + 1)
        
        # eval_results_dict['accloss/test/class/acc'] = test_acc
        # eval_results_dict['accloss/test/class/loss'] = test_loss
        
        confusion_matrix.to_tensorboard(
        writer, data_properties_dict[args.dataset].classes, epoch + 1, label='test/class/cm',)
        confusion_matrix.reset()
    else:
        # unsupervised model evaluation
        if args.linear_nn or args.exhaustive_test:
            # linear encoder is computationally expensive
            # TODO: the LeNet output size is hardcoded and needs to be inferred
            # TODO the linear classifier still needs a writer, try to remove
            train_loss, train_acc, test_loss, test_acc,  = train_linear_classifier(
                dataloader_train_eval,
                dataloader_test,
                84,
                data_properties_dict[args.dataset].n_classes,
                model=model,
                confusion_matrix=confusion_matrix,
                epochs=args.linear_nn_epochs,
                timestep=epoch + 1,
                test_every=args.linear_nn_test_every,
                writer=writer,
                description='object',
                device=device)
            # eval_results_dict['accloss/test/class/acc'] = test_acc
            # eval_results_dict['accloss/test/class/loss'] = test_loss
            # eval_results_dict['accloss/train/class/acc'] = train_acc
            # eval_results_dict['accloss/train/class/loss'] = train_loss
            
            writer.add_scalar(
                'accloss/test/class/accuracy', test_acc, epoch + 1)
            writer.add_scalar(
                'accloss/test/class/loss', test_loss, epoch + 1)
            writer.add_scalar(
                'accloss/train/class/accuracy', train_acc, epoch + 1)
            writer.add_scalar(
                'accloss/train/class/loss', train_loss, epoch + 1)
        
        if args.exhaustive_test:
            # reget the datasets and ignore the dataloaders, make sure to name the dataloaders something different
            _, _, _, dataset_train, dataset_train_eval, dataset_test = get_dataloaders(
            args, data_properties_dict)
            # 1) label by color
            dataset_train_eval.label_by = 'color'
            dataset_test.label_by = 'color'
            dl_train_eval = DataLoader(
                dataset_train_eval, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
            dl_test = DataLoader(
                dataset_test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
            # linear evaluation
            train_loss, train_acc, test_loss, test_acc = \
            train_linear_classifier(
                train_dataloader=dl_train_eval, 
                test_dataloader=dl_test, 
                input_features=84, 
                output_features=24, 
                model=model, 
                loss_fn=torch.nn.BCEWithLogitsLoss(),
                learning_rate=1e-3, #TODO: evaluate what to put in here as a default
                epochs=args.linear_nn_epochs, 
                timestep=epoch + 1, 
                test_every=args.linear_nn_test_every,
                label_type='n_hot', 
                confusion_matrix=None, 
                writer=writer, 
                description='color',
                device=device)
            writer.add_scalar(
                'accloss/test/color/accuracy', test_acc, epoch + 1)
            writer.add_scalar(
                'accloss/test/color/loss', test_loss, epoch + 1)
            writer.add_scalar(
                'accloss/train/color/accuracy', train_acc, epoch + 1)
            writer.add_scalar(
                'accloss/train/color/loss', train_loss, epoch + 1)
            
            # 2) label by intensity
            dataset_train_eval.label_by = 'power'
            dataset_test.label_by = 'power'
            dl_train_eval = DataLoader(
                dataset_train_eval, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
            dl_test = DataLoader(
                dataset_test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
            # linear evaluation
            train_loss, train_acc, test_loss, test_acc = \
            train_linear_regressor(
                train_dataloader=dl_train_eval,
                test_dataloader=dl_test,
                input_features=84,
                output_features=8,
                model=model,
                loss_fn=torch.nn.L1Loss(),
                learning_rate=1e-2, #TODO: evaluate what to put in here as a default
                epochs=args.linear_nn_epochs, 
                timestep=epoch + 1, 
                test_every=args.linear_nn_test_every,
                writer=writer,
                description='power',
                device=device)
            
            writer.add_scalar(
                'accloss/test/power/loss', test_loss, epoch + 1)
            writer.add_scalar(
                'accloss/train/power/loss', train_loss, epoch + 1)
            
            # 3) label by combined color intensity
            dataset_train_eval.label_by = 'lighting'
            dataset_test.label_by = 'lighting'
            dl_train_eval = DataLoader(
                dataset_train_eval, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
            dl_test = DataLoader(
                dataset_test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
            
            train_loss, train_acc, test_loss, test_acc = \
            train_linear_regressor(
                train_dataloader=dl_train_eval,
                test_dataloader=dl_test,
                input_features=84,
                output_features=24,
                model=model,
                loss_fn=torch.nn.L1Loss(),
                learning_rate=1e-1, #TODO: evaluate what to put in here as a default
                epochs=args.linear_nn_epochs, 
                timestep=epoch + 1, 
                test_every=args.linear_nn_test_every,
                writer=writer,
                description='lighting',
                device=device)
            
            writer.add_scalar(
                'accloss/test/lighting/loss', test_loss, epoch + 1)
            writer.add_scalar(
                'accloss/train/lighting/loss', train_loss, epoch + 1)
            pass
        
        # do the standard evaluation on the unsupervised representation
        features_train_eval, labels_train_eval = fuse_representations(
            model, dataloader_train_eval, device=device)
        features_test, labels_test = fuse_representations(
            model, dataloader_test, device=device)
        lstsq_model = lls_fit(features_train_eval,
                              labels_train_eval, data_properties_dict[args.dataset].n_classes, scipy=True)
        pred, test_acc = lls_eval(lstsq_model, features_test, labels_test)
        wb = wcss_bcss(features_test, labels_test, data_properties_dict[args.dataset].n_classes)
        pacmap_plot = get_pacmap(
            features_test, labels_test, epoch, data_properties_dict[args.dataset].n_classes, data_properties_dict[args.dataset].classes)
        
        print(
            f'Epoch {epoch + 1}: Read-Out Acc:{test_acc * 100:>6.2f}%, WCSS/BCSS:{wb:>8.4f}')
        
        # eval_results_dict['accloss/test/class/lstsq_accuracy'] = test_acc
        # eval_results_dict['analytics/test/class/WCSS-BCSS'] = wb
        
        writer.add_scalar(
            'accloss/test/class/lstsq_accuracy', test_acc, epoch + 1)
        writer.add_scalar(
            'analytics/test/class/WCSS-BCSS', wb, epoch + 1)
        writer.add_figure('test/class/PacMap', pacmap_plot, epoch + 1)
        
        confusion_matrix.update(pred.cpu(), labels_test.cpu())
        confusion_matrix.to_tensorboard(
            writer, data_properties_dict[args.dataset].classes, epoch + 1, label='test/class/lstsq_cm',)
        confusion_matrix.reset()
        
        # log_knn_acc(features_train_eval, labels_train_eval, features_test, labels_test, data_properties_dict[args.dataset].n_classes, data_properties_dict[args.dataset].classes,
        #             lambda x, x_pair: F.cosine_similarity(x.unsqueeze(1), x_pair.unsqueeze(0)), writer, epoch + 1, confusion_matrix, 'test/class', args.knn_batch_size)
        
        
        knn_pred, knn_acc = knn_eval(
            features_train_eval, labels_train_eval, features_test,
            labels_test, data_properties_dict[args.dataset].n_classes, lambda x, x_pair: F.cosine_similarity(x.unsqueeze(1), x_pair.unsqueeze(0), dim=2), args.knn_batch_size)
        
        
        writer.add_scalar(
            f'accloss/test/class/knn_accuracy', knn_acc, epoch + 1)
        
        confusion_matrix.update(knn_pred.cpu(), labels_test.cpu())
        confusion_matrix.to_tensorboard(
            writer, data_properties_dict[args.dataset].classes, epoch + 1, label=f'test/class/knn_cm',)
        confusion_matrix.reset()

    if args.save_embedding:
        writer.add_embedding(features_test, tag='Embedding', global_step=epoch + 1)
    

    model.train()
    
    return eval_results_dict

def log_to_tensorboard(eval_results_dict, writer, epoch):
    for k in eval_results_dict.keys():
        writer.add_scalar('accloss/test/class/accuracy', eval_results_dict[k], epoch)
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
