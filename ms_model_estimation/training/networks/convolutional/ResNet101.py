import torch
import torch.nn as nn
import torchvision
from ms_model_estimation.training.networks.convolutional.ResNet_CenterStriding import resnet101 as resnet101_center_striding
from ms_model_estimation.training.networks.convolutional.ResNet_CenterStriding import resnext101_32x8d as resnext101_32x8d_center_striding

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

        means = torch.Tensor([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        stds = torch.Tensor([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
        self.register_buffer('means', means)
        self.register_buffer('stds', stds)

    def forward(self, x):
        assert x.shape[1] == 3
        x = x - self.means
        x = x / self.stds
        return x

class ResNet101(nn.Module):

    def __init__(
            self, next=False, pretrained=True, fullyConv=True, centerStriding=True
    ):
        super(ResNet101, self).__init__()

        shownString = "Petrained" if pretrained else "NOT Pretrained"

        if not next and centerStriding:
            model = resnet101_center_striding(pretrained=pretrained, progress=True)
            print(f'Use ResNet101 with center striding and {shownString} model')
        elif not next and not centerStriding:
            model = torchvision.models.resnet101(pretrained=pretrained, progress=True)
            print(f'Use ResNet101 without center striding and {shownString} model')
        elif next and centerStriding:
            model = resnext101_32x8d_center_striding(pretrained=pretrained, progress=True)
            print(f'Use ResNet101 Next with center striding and {shownString} model')
        elif next and not centerStriding:
            model = torchvision.models.resnext101_32x8d(pretrained=pretrained, progress=True)
            print(f'Use ResNet101 Next without center striding and {shownString} model')
        else:
            raise Exception("Model is not defined.")

        if fullyConv:
            self.resnet101 = torch.nn.Sequential(*(list(model.children())[:-2]))
        else:
            # the model has the adaptive pooling
            self.resnet101 = torch.nn.Sequential(*(list(model.children())[:-1]))

        self.normalizeLayer = Normalize()

    def forward(self, x):

        x = self.normalizeLayer(x)
        return self.resnet101(x)
