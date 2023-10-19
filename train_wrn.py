from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import torchvision.transforms as transforms

import argparse

from architectures.blocks import ModernBasicBlock
from architectures.wide_resnet import WideResNet
from helpers.utils import running_average

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="wide resnet trainer")
# wandb args
parser.add_argument("--mode", default="disabled", type=str, choices=["online", "offline", "disabled"],
                    help="Wandb logging setting. Set to 'disabled' to prevent all logging (useful for debugging)")
# wrn args
parser.add_argument("--N", default=4, type=int, help="number of blocks per group; total number of convolutional layers "
                                                     "n = 6N + 4, so n = 28 -> N = 4")
parser.add_argument("--k", default=10, type=int, help="multiplier for the baseline width of each block; k=1 -> basic "
                                                      "resnet, k>1 -> wide resnet")
# training args
parser.add_argument("--batch", default=16, type=int, help="batch size")
parser.add_argument("--epochs", default=10, type=int, help="maximum epochs")
parser.add_argument("--lr", default=1e-5, type=float, help="learning rate")
parser.add_argument("--lr_schedule", default="60-120-160", type=str, help="set the schedule for when the learning "
                                                                             "rate should be dropped (SGD only). "
                                                                             "Format is '-' delimited epoch, e.g. "
                                                                             "60-120-160 by default")

USE_WANDB = False
if USE_WANDB:
    from helpers import wandb_functions

# ---------------------------HELPERS---------------------------------------------------------------------------------- #


def print_model_params(model):
    print("Number of parameters:")
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    print(pp)


def print_model_architecture(model):
    # Iterate over the model's parameters and print the information
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}, Parameters: {param.numel()}")
            total_params += param.numel()

    print(f"Total Trainable Parameters: {total_params}")

# ---------------------------DATA PREPARATION------------------------------------------------------------------------- #


def prepare_CIFAR10_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                             np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    train_transform = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transform
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch,
                                              shuffle=True, num_workers=6)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch,
                                             shuffle=False, num_workers=6)

    return trainloader, testloader

# ---------------------------MAIN PROGRAM----------------------------------------------------------------------------- #


def train(args, model, optimizer, trainloader, testloader, scheduler=None):

    for epoch in range(args.epochs):
        mean_loss = 0.0
        mean_acc = 0.0

        for i, data in enumerate(tqdm(trainloader)):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            # calculate statistics
            mean_loss = running_average(loss.item(), mean_loss, i)
            mean_acc = running_average(torch.sum(torch.argmax(outputs, dim=1) == labels) / labels.shape[0], mean_acc, i)

        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for j, data in enumerate(testloader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss = running_average(loss.item(), val_loss, j)
                val_acc = running_average(torch.sum(torch.argmax(outputs, dim=1) == labels) / labels.shape[0], val_acc, j)

        if scheduler:
            scheduler.step()
        tqdm.write(
            f"train loss: {mean_loss:.5f}; train acc: {mean_acc:.5f}, val loss: {val_loss:.5f}, val acc: {val_acc:.5f}"
        )
        if USE_WANDB:
            wandb_functions.wandb_log({
                "epoch": epoch,
                "train_loss": mean_loss,
                "train_acc": mean_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": scheduler.get_last_lr()[0] if scheduler else args.lr
            })

        print('Finished Training')


if __name__ == "__main__":
    args = parser.parse_args()

    configs = vars(args)
    configs['model_type'] = 'custom'

    if USE_WANDB:
        wandb_functions.wandb_init(configs, project="WRN-demo", mode=args.mode)

    # prepare CIFAR 10 data
    trainloader, testloader = prepare_CIFAR10_data()

    # forming Wide ResNet 28-10, WRN 28-10:
    n = 28
    N = int((n - 4) // 6)  # n=28 -> N=4
    k = 10
    print("Creating model WRN-{}-{} with N={}".format(n, k, N))

    model = WideResNet(ModernBasicBlock, [N, N, N], num_classes=10, k=k)
    model.to(device)
    print_model_params(model)
    #print_model_architecture(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        nesterov=True,
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )
    schedule = [int(milestone) for milestone in args.lr_schedule.split('-')]
    print(f"Dropping learning rates at epoch milestones: {schedule}")
    scheduler = MultiStepLR(optimizer, milestones=schedule, gamma=0.2)

    train(args, model, optimizer, scheduler, trainloader, testloader)

    if USE_WANDB:
        wandb_functions.wandb_finish()
