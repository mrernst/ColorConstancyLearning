#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import torch
import numpy as np
import matplotlib as mpl
import re
import itertools
from textwrap import wrap

# custom functions
# -----


def images_to_probs(output, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    # output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())

    return preds, [torch.nn.functional.softmax(el, dim=0)[i].item()
                   for i, el in zip(preds, output)]


def plot_classes_preds(output, images, labels, classes):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    _, channels, height, width = images.shape

    one_channel = True if channels in [1, 2] else False
    stereo = True if (channels % 2) == 0 else False

    preds, probs = images_to_probs(output, images)
    # plot the images in the batch, along with predicted and true labels
    fig = mpl.figure.Figure(
        figsize=(12, 12), dpi=90, facecolor='w', edgecolor='k')
    total_imgs = len(images) if len(images) < 10 else 10
    for idx in np.arange(total_imgs):
        ax = fig.add_subplot(5, 5, idx+1, xticks=[], yticks=[])
        img = images[idx]
        if stereo:
            # img = img.view(channels//2,height*2,width)
            img1, img2 = torch.split(img, channels//2)
            img = torch.cat([img1, img2], dim=1)
        elif one_channel:
            img = img.mean(dim=0)
            img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()

        if one_channel:
            if len(npimg.shape) > 2:
                npimg = np.transpose(npimg, (1, 2, 0))[:, :, 0]
            ax.imshow(npimg, cmap="Greys")
        else:
            ax.imshow(np.transpose(npimg, (1, 2, 0)))

        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
            color=("green" if preds[idx] == labels[idx].item() else "red"),
            fontsize=6)

    return fig


# custom classes
# -----
class ConfusionMatrix(object):
    """
    Holds and updates a confusion matrix object given the networks
    outputs
    """

    def __init__(self, n_cls):
        self.n_cls = n_cls
        self.reset()

    def reset(self):
        self.val = torch.zeros(self.n_cls, self.n_cls, dtype=torch.float32)

    def update(self, batch_output, batch_labels):
        _, topi = batch_output.topk(1)
        # this is changed to use the predictions of the lls rightaway
        oh_labels = torch.nn.functional.one_hot(batch_labels, self.n_cls)
        oh_outputs = torch.nn.functional.one_hot(
            topi, self.n_cls).view(-1, self.n_cls)
        self.val += torch.matmul(torch.transpose(oh_labels, 0, 1), oh_outputs)

    def print_misclassified_objects(self, encoding, n_obj=5):
        """
        prints out the n_obj misclassified objects given a
        confusion matrix array cm.
        """
        cm = self.val.numpy()
        encoding = np.array(encoding)

        np.fill_diagonal(cm, 0)
        maxind = self.largest_indices(cm, n_obj)
        most_misclassified = encoding[maxind[0]]
        classified_as = encoding[maxind[1]]
        print('most misclassified:', most_misclassified)
        print('classified as:', classified_as)
        pass

    def largest_indices(self, arr, n):
        """
        Returns the n largest indices from a numpy array.
        """
        flat_arr = arr.flatten()
        indices = np.argpartition(flat_arr, -n)[-n:]
        indices = indices[np.argsort(-flat_arr[indices])]
        return np.unravel_index(indices, arr.shape)

    def to_figure(self, labels, title='Confusion matrix',
                  normalize=False,
                  colormap='Oranges'):
        """
        Parameters:
            confusion_matrix                : Confusionmatrix Array
            labels                          : This is a list of labels which will
                be used to display the axis labels
            title = 'confusion matrix'        : Title for your matrix
            tensor_name = 'MyFigure/image'  : Name for the output summary tensor
            normalize = False               : Renormalize the confusion matrix to
                ones
            colormap = 'Oranges'            : Colormap of the plot, Oranges fits
                with tensorboard visualization


        Returns:
            summary: TensorFlow summary

        Other items to note:
            - Depending on the number of category and the data , you may have to
                modify the figsize, font sizes etc.
            - Currently, some of the ticks dont line up due to rotations.
        """
        cm = self.val
        if normalize:
            cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm, copy=True)
            cm = cm.astype('int')

        np.set_printoptions(precision=2)
        import matplotlib.pyplot as plt
        #fig = mpl.figure.Figure(
        fig = plt.figure(
            figsize=(14, 10), dpi=90, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        im = ax.imshow(cm, cmap=colormap)
        fig.colorbar(im)

        classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x)
                   for x in labels]
        classes = ['\n'.join(wrap(l, 40)) for l in classes]

        tick_marks = np.arange(len(classes))

        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=6, rotation=-90,  ha='center')
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()

        ax.set_ylabel('True Label', fontsize=8)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes, fontsize=6, va='center')
        ax.yaxis.set_label_position('left')
        ax.yaxis.tick_left()

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, format(cm[i, j], '.0f') if cm[i, j] != 0 else '.',
                    horizontalalignment="center", fontsize=6,
                    verticalalignment='center', color="black")
        fig.set_tight_layout(True)
        return fig

    def to_tensorboard(self, writer, class_encoding, global_step,  label='confusionmatrix', colormap='Oranges'):

        writer.add_figure(label, self.to_figure(
            class_encoding, colormap=colormap), global_step=global_step)

        #writer.close()


class PrecisionRecall(object):
    """
    Holds and updates values for precision and recall object
    """

    def __init__(self, n_cls):
        self.n_cls = n_cls
        self.reset()

    def reset(self):
        self.probabilities = []
        self.predictions = []
        # self.labels = []

    def update(self, batch_output, batch_labels):
        _, topi = batch_output.topk(1)
        class_probs_batch = [torch.nn.functional.softmax(
            el, dim=0) for el in batch_output]

        self.probabilities.append(class_probs_batch)
        self.predictions.append(torch.flatten(topi))
        # self.labels.append(batch_labels)

    def to_tensorboard(self, writer, class_encoding, global_step):
        '''
        Takes in a "the class_encoding" i.e. from 0 to 9 and plots the
        corresponding precision-recall curves to tensorboard
        '''

        probs = torch.cat([torch.stack(b)
                           for b in self.probabilities]).view(-1, self.n_cls)
        preds = torch.cat(self.predictions).view(-1)
        # labels = torch.cat(self.labels).view(-1)

        for class_index, class_name in enumerate(class_encoding):

            # subset = np.where(labels == class_index)
            # sub_probs = probs[subset[0]]
            # sub_preds = preds[subset[0]]
            #
            # ground_truth = sub_preds == class_index
            # probability = sub_probs[:, class_index]

            ground_truth = preds == class_index
            probability = probs[:, class_index]

            writer.add_pr_curve(class_encoding[class_index],
                                ground_truth,
                                probability,
                                global_step=global_step)

        #writer.close()

class PacMap(object):
    # TODO: write pacmap class instead of get_pacmap function
    # __init__
    # calculate
    # to_figure
    # to_tensorboard
    pass

# ----------------
# main program
# ----------------
if __name__ == "__main__":
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
