import torch.nn as nn
import torch.nn.functional as F
import torch


# Because of the standard, vgg model can have different layers
# However, the size of cifar10 picture is too small. Hence, vgg16 and vgg19 are not used in this model
# This dictionary contains all parameter for vgg11 and vgg13
cfgs = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
}


def make_features(cfg: list):  # This function use the parameter from cfgs dictionary to create structure of model
    layers = []
    in_channels = 3
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                       nn.ReLU(True)]
            in_channels = x
    return nn.Sequential(*layers)

class VGG(nn.Module):
    def __init__(self, features, class_num=10):
        super(VGG, self).__init__()
        self.features = features
        # 3 fully connected layers with dropout
        self.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(512*2*2, 2048),
            nn.ReLU(True),
            nn.Linear(2048,2048),
            nn.ReLU(True),
            nn.Linear(2048,class_num)
    )

    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

def vgg(model_name, **kwargs):
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict! You can try VGG11 or VGG13".format(model_name))
        exit(-1)
    model = VGG(make_features(cfg), **kwargs)
    return model
