#!/usr/bin/python
# _____________________________________________________________________________

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import torchvision.models as models
from torch import nn


# configuration module
# ------
from config import args


# ----------------
# networks
# ----------------   


class ResNet18(nn.Module):
    def __init__(self, num_classes=50):
        super(ResNet18, self).__init__()
        # regular resnet 18 as feature extractor
        resnet18 = models.resnet18(weights=None)
        # setting the final layer followed by an MLP to be the representation
        # network (encoder)
        setattr(resnet18, 'fc', nn.Linear(512, args.hidden_dim))
        self.encoder = nn.Sequential(resnet18, nn.BatchNorm1d(args.hidden_dim),
            nn.ReLU(), nn.Linear(256, args.feature_dim, bias=False))
        # a linear layer as projection network
        self.projector = MLPHead(args.feature_dim, args.hidden_dim, args.feature_dim)
        self.linear_out = nn.Linear(args.feature_dim, num_classes)
    
    def forward(self, x):
        """
        x: image tensor of (B x 3 x 64 x 64)
        return: representation tensor h (B x FEATURE_DIM), projection tensor z
        (B x HIDDEN_DIM) that should be used by the loss function.
        """
        representation = self.encoder(x)
        
        if args.main_loss == 'supervised':
            projection = self.linear_out(representation)
        elif args.projectionhead:
            projection = self.projector(representation)
        else:
            projection = representation
        
        return representation, projection


class LeNet5(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        self.layer3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        
        self.encoder = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3
        )
        
        self.projector = MLPHead(84, args.hidden_dim, args.feature_dim)
        self.linear_out = nn.Linear(84, num_classes)

        
    def forward(self, x):
        representation = self.encoder(x)
        
        if args.main_loss == 'supervised':
            projection = self.linear_out(representation)
        elif args.projectionhead:
            projection = self.projector(representation)
        else:
            projection = representation
        
        return representation, projection


class AlexNet(nn.Module):
    def __init__(self, num_classes=50):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            #nn.Linear(9216, 4096),
            nn.Linear(256, 128),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            #nn.Linear(4096, 4096),
            nn.ReLU())
        self.encoder = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            self.fc,
            self.fc1,
        )
        self.linear_out= nn.Sequential(
            nn.Linear(128, num_classes))
            #nn.Linear(4096, num_classes))
        self.projector = MLPHead(128, args.hidden_dim, args.feature_dim)
        
        
    def forward(self, x):
        representation = self.encoder(x)
        
        if args.main_loss == 'supervised':
            projection = self.linear_out(representation)
        elif args.projectionhead:
            projection = self.projector(representation)
        else:
            projection = representation
        
        return representation, projection

class SplitOutput(nn.Module):
    def __init__(self, return_split=0):
       super().__init__()
       self.return_split = return_split
    
    def forward(self,x):
        return x[self.return_split]

class LinearClassifier(nn.Module):
   def __init__(self, num_features=128, num_classes=50):
       super().__init__()
       self.linear_out = nn.Sequential(
           nn.Flatten(),
           nn.Linear(num_features, num_classes),
           #nn.ReLU(),
       )   
       
   def forward(self, x):
       pred = self.linear_out(x)
       return pred


class MLPHead(nn.Module):
        def __init__(self, in_channels, mlp_hidden_size, projection_size):
            super(MLPHead, self).__init__()
        
            self.net = nn.Sequential(
                nn.Linear(in_channels, mlp_hidden_size),
                nn.BatchNorm1d(mlp_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_size, projection_size)
            )
        
        def forward(self, x):
            return self.net(x)


#[channels, kernel size, stride, padding] structure: [64, 8, 4, 2], [128, 4, 2, 1], [256, 2, 2, 1], [256, 2, 2, 1]. These are followed by an average pooling layer and a linear layer ending with $20$ units



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