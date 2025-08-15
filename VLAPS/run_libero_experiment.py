
import os

import jax
print("JAX devices before Libero import:", jax.devices())

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import logging

import wandb
import datetime
# import torch

import yaml

import numpy as np

from pathlib import Path

from vlaps.environments.libero_runner import LiberoSingleEpisodeRunner, LiberoMultiEpisodeRunner

@hydra.main(version_base=None, config_path="config", config_name="config_vlaps_octo")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    #####################################
    # Setup the wandb experiment logger #
    #####################################
    now = datetime.datetime.now()
    datetime_exp_name = now.strftime(
        "%Y-%m-%d_%H-%M-%S_" + cfg.experiment_name
    )

    # Set Wandb to run in offline mode
    os.environ["WANDB_MODE"] = cfg.wandb.wandb_mode

    run = wandb.init(
        project=cfg.project_name,
        entity="real-lab",
        name=datetime_exp_name,
        tags=cfg.tags,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=cfg.wandb.dir
    )
    run_dir = wandb.run.dir

    wandb.log({"config": OmegaConf.to_container(cfg, resolve=True)})

    # Save a yaml copy of the config to the wandb run.
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path, "w") as f:
        yaml.safe_dump(OmegaConf.to_container(cfg, resolve=True), f)

    with open_dict(cfg):
        cfg.datetime_exp_name = datetime_exp_name
        
        if 'seed' not in cfg:
            cfg.seed = int(now.strftime("%Y%m%d%H%M%S"))

    ###############
    #   Set seed  #
    ###############
    logging.info(f"Seed: {cfg.seed}")
    np.random.seed(cfg.seed)

    ###############################
    # Create an experiment runner #
    ###############################
    runner = LiberoMultiEpisodeRunner(
        cfg,
        seed=cfg.seed,
        run_name=datetime_exp_name,
    )

    ######################
    # Run the experiment #
    ######################
    runner.run()

    ##################
    # End the experiment #
    ##################
    wandb.finish()

if __name__ == "__main__":
    main()