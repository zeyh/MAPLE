import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.models.resnet import resnet50
from torch import Tensor
from typing import Tuple


class Model(nn.Module):
    def __init__(self, input_dim: int = None, projector_dims: list = [512, 128], use_resnet: bool = False, dataset: str = "", image_shape: tuple = None):
        super(Model, self).__init__()

        self.use_resnet = use_resnet
        self.dataset = dataset
        self.image_shape = image_shape

        encoder_output_dim = 2048  

        if use_resnet:
            from torchvision.models.resnet import resnet50
            # ResNet encoder
            self.f = []
            for name, module in resnet50().named_children():
                if name == "conv1":
                    # module = nn.Conv2d(
                    #     3, 64, kernel_size=3, stride=1, padding=1, bias=False
                    # )
                    # Determine input channels based on image_shape
                    if self.image_shape and len(self.image_shape) >= 3:
                        input_channels = self.image_shape[0]  # First dimension is channels
                    else:
                        input_channels = 3  # Default to 3 channels
                    
                    module = nn.Conv2d(
                        input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
                    )
                if dataset == "cifar10" or dataset == "cifar100":
                    if not isinstance(module, nn.Linear) and not isinstance(
                        module, nn.MaxPool2d
                    ):
                        self.f.append(module)
                elif dataset == "stl10":
                    if not isinstance(module, nn.Linear):
                        self.f.append(module)
            self.f = nn.Sequential(*self.f)
        else:
            # Linear encoder (current implementation)
            if input_dim is None:
                raise ValueError("input_dim must be specified for linear encoder")
            self.f = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, encoder_output_dim),
                nn.ReLU()
            )

        projector_dims = [encoder_output_dim] + projector_dims # [2048, 512, 128]
            
        layers = []
        for i in range(len(projector_dims) - 2):
            layers.append(
                nn.Linear(projector_dims[i], projector_dims[i + 1], bias=False)
            )
            layers.append(nn.BatchNorm1d(projector_dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(projector_dims[-2], projector_dims[-1], bias=False))
        self.g = nn.Sequential(*layers)

        # self.g = nn.Sequential(
        #     nn.Linear(2048, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Linear(512, 128)
        # )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if self.use_resnet and len(x.shape) == 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, *self.image_shape)

        x = self.f(x)
        if len(x.shape) > 2:
            x = torch.flatten(x, start_dim=1) 
        feature = x
        out = self.g(feature)

        return feature, out
