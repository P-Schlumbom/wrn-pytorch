# An object-oriented pytorch implementation of Wide Residual Networks

Based on the paper "Wide Residual Networks" (https://arxiv.org/abs/1605.07146) and the corresponding code at: https://github.com/szagoruyko/wide-residual-networks


The original code is nice, but uses an odd approach to implementing the Wide ResNet (WRN) architecture. 
Here it is re-implemented using the standard object-oriented pytorch approach for a more legible and adaptable version.
Included is a demo script for training the OO version, as well as the original code, so that they can be directly compared.
The new OO version should achieve the same validation performance 
(96% validation accuracy with WRN-28-10 on the CIFAR10 dataset after 200 epochs of training).

## Training
Simply run `train_wrn.py` with the default parameters. The parameters are:

- **N (int; default 4)** number of blocks per group; total number of convolutional layers n = 6N + 4, so n = 28 -> N = 4
- **k (int; default 10)** multiplier for the baseline width of each block; k=1 -> basic resnet, k>1 -> wide resnet
- **batch (int; default 128)** the batch size
- **epochs (int; default 200)** the number of epochs to train for
- **lr (float; default 0.1)** the learning rate to use
- **lr_schedule (string; default "60-120-160:")** the epochs at which to reduce the learning rate (by 0.2)
- **mode (string; default "disabled")** wandb parameter; set to 'online' to enable logging (requires a wandb account).

This sets up a WRN-28-10 architecture exactly as described in the original paper, and trains it according to exactly the same training regime.

### using wandb
It's always annoying when a github project uses libraries you don't actually need and perhaps don't even want to use, 
so I've put all the wandb calls into a separate script. In `train_wrn.py` the variable `USE_WANDB` is set to `False`
by default, in which case the `wandb_functions.py` script should never even be called and you don't have to worry about it.
However if you set `--mode` to anything other than `disabled` it will try to use `wandb`...