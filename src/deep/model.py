
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes, weights: str=None):

    model = models.resnet50(pretrained=False)

    if weights is not None:
        model.fc = nn.Identity()
        state_dict = torch.load(weights, map_location='cpu')
        model.load_state_dict(state_dict)

    model.fc = nn.Linear(2048, num_classes)
    return model

def get_optimizer(model, lr, momentum, iterations, rel_milestones):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    milestones = [int(iterations*milestone) for milestone in rel_milestones]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)
    return optimizer, scheduler
