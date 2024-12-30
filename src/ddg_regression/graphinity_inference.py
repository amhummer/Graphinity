import torch
import pytorch_lightning as pl

import yaml
from collections import defaultdict
from pathlib import Path
import argparse

from models.egnn_model import ddgEGNN
from base.dataset import ddgDataSet

#import warnings
#warnings.simplefilter(action="ignore", category=FutureWarning)


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

    # Load model weights from checkpoint specified in config file
    checkpoint = torch.load(config["initialize_weights"]["checkpoint_file"], map_location=device)
    pretrained_dict = {k: v for k, v in checkpoint["state_dict"].items()}
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"Loaded model weights from {config['initialize_weights']['checkpoint_file']}")

    # set up model training
    trainer = pl.Trainer(
        **config["trainer_params"])

    # test model
    if config["test"]:
        model.test_set_predictions = []
        trainer.test(model)
        model.save_test_predictions(Path(config["save_dir"]) / f"preds_{config['name']}.csv")


if __name__ == "__main__":
    # parse config file
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    with open(args.config) as yaml_file_handle:
        config = yaml.safe_load(yaml_file_handle)
    config = defaultdict(lambda: None, config)

    main(config)

