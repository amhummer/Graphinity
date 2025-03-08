import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb

import yaml
from collections import defaultdict
from pathlib import Path
import argparse
import shutil

from models.egnn_model import ddgEGNN
from base.dataset import ddgDataSet


def main(config: dict):
    """ Build EGNN model using params specified in config file """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config["model"] == "ddgEGNN":
        ModelClass = ddgEGNN
    else:
        raise NotImplementedError

    # Set up model with parameters specified in config file
    model = ModelClass(
            dataset_config=config["dataset_params"],
            loader_config=config["loader_params"],
            trainer_config=config["trainer_params"],
            **config["model_params"]
        )

    # Options to load model checkpoints
    ## Restore
    if config["restore"] is not None:
        model = ModelClass.load_from_checkpoint(
            config["restore"],
            dataset_config=config["dataset_params"],
            loader_config=config["loader_params"],
            trainer_config=config["trainer_params"],
            **config["model_params"])
    ## Load weights
    if config["initialize_weights"] is not None:
        checkpoint = torch.load(config["initialize_weights"]["checkpoint_file"], map_location=device)
        pretrained_dict = {k: v for k, v in checkpoint["state_dict"].items()}
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded model weights from {config['initialize_weights']['checkpoint_file']}")

    # Define trainer
    ## set up wandb (Weights & Biases) logger
    trainer_config = config["trainer_params"]
    if config["logger_params"]["wandb_bool"]:
        logger = WandbLogger(
            save_dir=Path(config["save_dir"]),
            offline=False,
            project=f"{config['logger_params']['wandb']}",
            name=f"{config['name']}",
            group=config["logger_params"]["group"],
        )
        logger.log_hyperparams({
                "graph_generation_mode": config["dataset_params"]["graph_generation_mode"],
                **config["model_params"],
            })
    else:
        logger = TensorBoardLogger(save_dir=config["save_dir"])

    restore = config["restore"] if "restore" in config else None

    checkpoint_callback = ModelCheckpoint(
        monitor="val_pearson_corr",
        mode="max",
    )

    # Set up model training
    trainer = pl.Trainer(
        default_root_dir=config["save_dir"],
        logger=logger,
        resume_from_checkpoint=restore,
        callbacks=[checkpoint_callback],
        **trainer_config)

    # Train model
    if config["train"]:
        trainer.fit(model)

    # Test model
    if config["test"]:
        model.test_set_predictions = []
        trainer.test(model)
        model.save_test_predictions(Path(config["save_dir"]) / f"preds_{config['name']}.csv")

    # Save final model parameters
    ## NB will also save checkpoints with best validation Pearson's correlation
    torch.save({
        "epoch": config["trainer_params"]["max_epochs"],
        "model_state_dict": model.state_dict(),
        }, config["save_dir"] + f"checkpoint_final_{config['name']}.pt")


if __name__ == "__main__":
    # parse config file
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    with open(args.config) as yaml_file_handle:
        config = yaml.safe_load(yaml_file_handle)
    config = defaultdict(lambda: None, config)

    # Save copy of config file for future reference
    config["save_dir"] = config["save_dir"] + config["name"] + "/"
    Path(config["save_dir"]).mkdir(exist_ok=True, parents=True)

    shutil.copyfile(args.config, config["save_dir"] + f"config_{config['name']}.yaml")

    main(config)
