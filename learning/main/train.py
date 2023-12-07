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
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms

# configuration module
# -----

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
#import config
from config import args

# custom libraries
# -----
from utils.general import save_model, load_model, save_args
from utils.datasets import SimpleTimeContrastiveDataset
from utils.networks import ResNet18, LeNet5, AlexNet, MLPHead
from utils.losses import RELIC_TT_Loss, VICReg_TT_Loss, SimCLR_TT_Loss, Decorrelation_TT_Loss
from utils.augmentations import get_transformations, TwoContrastTransform

from utils.evaluation import get_representations, lls_fit, lls_eval, supervised_eval, knn_eval, wcss_bcss, get_pacmap, train_linear_classifier
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
RUN_NAME = f'{datetime.datetime.now().strftime("%d-%m-%y_%H:%M")}_{args.name}_seed_{args.seed}_{args.dataset}_aug_{args.contrast}_{args.main_loss}_reg_{args.reg_loss}'
LR_DECAY_EPOCHS = [0]

# similarity functions dictionary
SIMILARITY_FUNCTIONS = {
    'cosine': lambda x, x_pair: F.cosine_similarity(x.unsqueeze(1), x_pair.unsqueeze(0), dim=2),
    'RBF': lambda x, x_pair: -torch.cdist(x, x_pair)
}

# loss dictionary for different losses
MAIN_LOSS = {
    'SimCLR': SimCLR_TT_Loss(SIMILARITY_FUNCTIONS[args.similarity], args.batch_size, args.temperature, device),
    'VICReg': VICReg_TT_Loss(),
    'supervised': lambda x, x_pair, labels: F.cross_entropy(x, labels),
    'supervised_representation': lambda x, x_pair, labels: F.cross_entropy(x, labels),
}

REG_LOSS = {
    'RELIC': RELIC_TT_Loss(SIMILARITY_FUNCTIONS[args.similarity]),
    'Decorrelation': Decorrelation_TT_Loss(args.hidden_dim, device)
}
DATASETS = {
    'C3': {'class': SimpleTimeContrastiveDataset,
           'size': 15000,
           'rgb_mean': (0.7709, 0.7642, 0.7470),  # need to calculate this
           'rgb_std': (0.0835, 0.0842, 0.0840),
           },
}

# define manual random seed
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
# custom function
# -----


def log_knn_acc(train_features, train_labels, test_features, test_labels, test_dataset, sim_function, tb_writer, timestep, confusion_matrix, qualifier_str='test/class', batch_size=512):
    knn_pred, knn_acc = knn_eval(train_features, train_labels, test_features,
                                 test_labels, test_dataset.n_classes, sim_function, batch_size)

    tb_writer.add_scalar(
        f'accloss/{qualifier_str}/knn_accuracy', knn_acc, timestep)
    confusion_matrix.update(knn_pred.cpu(), test_labels.cpu())
    confusion_matrix.to_tensorboard(
        tb_writer, test_dataset.classes, timestep, label=f'{qualifier_str}/knn_cm',)
    confusion_matrix.reset()
    pass


def log_lls_acc():
    pass


def train():
    print(f"[INFO:] Using {device} backend")
    path_to_experiment = os.path.join(args.log_dir, RUN_NAME)
    # make directory
    os.makedirs(path_to_experiment, exist_ok=True)
    # write config to a json file
    save_args(path_to_experiment, args.__dict__)

    # prepare tensorboard writer
    writer = SummaryWriter(log_dir=path_to_experiment)

    # get transformations for validation and for training
    train_transform, val_transform = get_transformations(
        contrast_type=args.contrast,
        rgb_mean=DATASETS[args.dataset]['rgb_mean'],
        rgb_std=DATASETS[args.dataset]['rgb_std'],
        crop_size=args.crop_size)
    
    dataset_train = SimpleTimeContrastiveDataset(
        root=os.path.expanduser(args.data_root) + f'/{args.dataset}/{args.train_split}',
        transform=train_transform,
        contrastive=True if (args.contrast == 'time' or
                             'combined' in args.contrast) else False,
    )
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                  num_workers=args.num_workers, shuffle=True, drop_last=True)

    dataset_train_eval = SimpleTimeContrastiveDataset(
        root=os.path.expanduser(args.data_root) + f'/{args.dataset}/{args.train_split}',
        transform=val_transform,
        contrastive=False
    )
    
    dataloader_train_eval = DataLoader(
        dataset_train_eval, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    dataset_test = SimpleTimeContrastiveDataset(
        root=os.path.expanduser(args.data_root) + f'/{args.dataset}/{args.test_split}',
        transform=val_transform,
        contrastive=False
    )
    dataloader_test = DataLoader(
        dataset_test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    if args.encoder == 'resnet18':
        net = ResNet18(num_classes=dataset_train.n_classes).to(device)
    elif args.encoder == 'LeNet5':
        net = LeNet5(num_classes=dataset_train.n_classes).to(device)
    elif args.encoder == 'AlexNet':
        net = AlexNet(num_classes=dataset_train.n_classes).to(device)

    else:
        raise NotImplementedError(
            '[INFO] Specified Encoder is not implemented')

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lrate, weight_decay=args.weight_decay)

    # get initial result and save plot and record

    if args.main_loss == 'supervised':

        acc, test_loss, class_cm = supervised_eval(
            net, dataloader_test, F.cross_entropy, dataset_test.n_classes, device=device)
        writer.add_scalar('accloss/test/class/accuracy', acc, 0)
        writer.add_scalar('accloss/test/class/loss', test_loss, 0)
        class_cm.to_tensorboard(
            writer, dataset_test.classes, 0, label='test/class/cm',)
    else:

        if args.linear_nn:
            # linear encoder only on the test set?
            _, acc = train_linear_classifier(dataloader_train_eval, dataloader_test, 84, dataset_train_eval.n_classes,
                                             model=net, epochs=100, global_epoch=0, test_every=1, writer=writer, device=device)
                                             
        # maybe we should do the evaluation in cpu space instead
        features_train_eval, labels_train_eval = get_representations(
            net, dataloader_train_eval, device=device)
        features_test, labels_test = get_representations(
            net, dataloader_test, device=device)
        lstsq_model = lls_fit(features_train_eval,
                              labels_train_eval, dataset_train_eval.n_classes)
        pred, acc = lls_eval(lstsq_model, features_test, labels_test)
        wb = wcss_bcss(features_test, labels_test, dataset_test.n_classes)
        pacmap_plot = get_pacmap(
            features_test, labels_test, 0, dataset_test.n_classes, dataset_test.classes)

        print(
            f'Initial result: Read-Out Acc:{acc * 100:>6.2f}%, WCSS/BCSS:{wb:>8.4f}')
        writer.add_scalar('accloss/test/class/accuracy', acc, 0)
        writer.add_scalar('analytics/test/class/WCSS-BCSS', wb, 0)
        writer.add_figure('test/class/PacMap', pacmap_plot, 0)

        class_cm = ConfusionMatrix(n_cls=dataset_test.n_classes)
        class_cm.update(pred.cpu(), labels_test.cpu())
        class_cm.to_tensorboard(
            writer, dataset_test.classes, 0, label='test/class/cm',)
        class_cm.reset()

        log_knn_acc(features_train_eval, labels_train_eval, features_test, labels_test, dataset_test,
                    SIMILARITY_FUNCTIONS[args.similarity], writer, 0, class_cm, 'test/class', args.knn_batch_size)

        if args.save_embedding:
            writer.add_embedding(features_test, tag='Embedding', global_step=0)

    # decrease learning rate by a factor of 0.3 every 10 epochs
    # scheduler = StepLR(optimizer, 10, 0.3)
    if args.cosine_decay:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.n_epochs,
                                      eta_min=args.lrate * (args.lrate_decay ** 3))
    elif args.exp_decay:
        scheduler = ExponentialLR(optimizer, 1.0)
    else:
        scheduler = StepLR(optimizer, 10, args.lrate_decay)

    epoch_loop = tqdm(range(args.n_epochs), ncols=80)
    for epoch in epoch_loop:
        num_batches = len(dataloader_train)
        sum_epoch_loss = 0
        for (x, x_pair), labels in dataloader_train:
            x, y = torch.cat([x, x_pair], 0).to(device), labels.to(device)
            representation, projection = net(x)
            projection, pair = projection.split(projection.shape[0]//2)
            loss = MAIN_LOSS[args.main_loss](projection, pair, y)
            if args.reg_loss:
                if args.reg_loss == 'Decorrelation':
                    representation = representation.split(
                        projection.shape[0]//2)[0].T
                    loss += args.decorr_weight * \
                        REG_LOSS[args.reg_loss](representation, pair)
                else:
                    loss += REG_LOSS[args.reg_loss](projection, pair)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loop.set_description(f'Loss: {loss.item():>8.4f}')
            sum_epoch_loss += loss.item()
            # always write down the loss
        writer.add_scalar('accloss/train/loss',
                          sum_epoch_loss/num_batches, epoch + 1)
        net.train()

        # update learning rate
        lr_decay_steps = torch.sum(epoch > torch.Tensor(LR_DECAY_EPOCHS))
        if (lr_decay_steps > 0 and not (args.cosine_decay or args.exp_decay)):
            scheduler.gamma = args.lrate_decay ** lr_decay_steps
        scheduler.step()

        if args.save_model:
            if (epoch + 1) % args.save_every == 0 or (epoch+1) == args.n_epochs:
                save_model(net, writer, epoch + 1)

        if (epoch+1) % args.test_every == 0 or (epoch+1) == args.n_epochs:
            # set network to evaluation mode
            net.eval()

            if args.main_loss == 'supervised':

                writer.add_scalar('analytics/learningrate',
                                  scheduler.get_last_lr()[0], epoch + 1)

                acc, test_loss, class_cm = supervised_eval(
                    net, dataloader_test, F.cross_entropy, dataset_test.n_classes, device=device)
                writer.add_scalar(
                    'accloss/test/class/accuracy', acc, epoch + 1)
                writer.add_scalar('accloss/test/class/loss',
                                  test_loss, epoch + 1)
                class_cm.to_tensorboard(
                    writer, dataset_test.classes, epoch + 1, label='test/class/cm',)

            else:

                if args.linear_nn:
                    _, acc = train_linear_classifier(dataloader_train_eval, dataloader_test, 84, dataset_train_eval.n_classes,
                                                     model=net, epochs=100, global_epoch=epoch, test_every=1, writer=writer, device=device)

                    # in this linear classifier function also the confusion matrix and the wcss-bcss should be included, ideally in an iterative way. Then the lstsq_model only is needed for online testing
                    # also you need some testing here to actually see when learning converges

                features_train_eval, labels_train_eval = get_representations(
                    net, dataloader_train_eval, device=device)
                features_test, labels_test = get_representations(
                    net, dataloader_test, device=device)
                lstsq_model = lls_fit(
                    features_train_eval, labels_train_eval, dataset_train_eval.n_classes)
                pred, acc = lls_eval(lstsq_model, features_test, labels_test)
                wb = wcss_bcss(features_test, labels_test,
                               dataset_test.n_classes)
                pacmap_plot = get_pacmap(features_test, labels_test, epoch + 1,
                                         dataset_test.n_classes, dataset_test.classes)
                print(f"Method: {RUN_NAME.split('~')[0]}, Epoch: {epoch + 1}, "
                      f"Read-Out Acc:{acc * 100:>6.2f}%, WCSS/BCSS:{wb:>8.4f}")

                # record results
                writer.add_scalar(
                    'accloss/test/class/accuracy', acc, epoch + 1)
                writer.add_scalar(
                    'analytics/test/class/WCSS-BCSS', wb, epoch + 1)
                writer.add_scalar('analytics/learningrate',
                                  scheduler.get_last_lr()[0], epoch + 1)
                writer.add_figure('test/class/PacMap', pacmap_plot, epoch + 1)

                class_cm.update(pred.cpu(), labels_test.cpu())
                class_cm.to_tensorboard(
                    writer, dataset_test.classes, epoch + 1, label='test/class/cm',)
                class_cm.reset()

                log_knn_acc(features_train_eval, labels_train_eval, features_test, labels_test, dataset_test,
                            SIMILARITY_FUNCTIONS[args.similarity], writer, epoch + 1, class_cm, 'test/class', args.knn_batch_size)

            # set network back to training mode
            net.train()

            if args.save_embedding:
                writer.add_embedding(
                    features_test, tag='Embedding', global_step=epoch + 1)


# ----------------
# main program
# ----------------

if __name__ == '__main__':
    for i in range(args.n_repeat):
        RUN_NAME = RUN_NAME.rsplit('~')[0] + f'~{i}'
        train()

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
