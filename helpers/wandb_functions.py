"""
Contains all wandb functionality; simply don't call this script if you don't want to use wandb (weights and biases)
"""

import wandb


def wandb_init(configs, project="WRN-demo", mode="disabled"):
    wandb.init(
        project=project,
        config=configs,
        mode=mode
    )


def wandb_log(data):
    wandb.log(data)


def wandb_finish():
    wandb.finish()

