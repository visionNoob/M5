# Templete by https://github.com/victoresque/pytorch-template
# and https://github.com/microsoft/LightGBM/blob/master/examples/python-guide/sklearn_example.py#L32
# and https://github.com/skorch-dev/skorch
import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random
import argparse
import collections
import lightgbm as lgb
from models.model_factory import get_network
from multiprocessing import Pool
from parse_config import ConfigParser
from data.util import seed_everything
from data.dataset_factory import get_dataset
from sklearn.metrics import mean_squared_error
import time
import wandb
from wandb.lightgbm import wandb_callback

warnings.filterwarnings("ignore")


def main(config):
    print(config["dstpath"])
    logger = config.get_logger("train")
    SEED = config["seed"]
    model = get_network(config)
    ORIGINAL = config["data_loader"]["path_original"]
    STORES_IDS = pd.read_csv(ORIGINAL + "sales_train_validation.csv")["store_id"]
    STORES_IDS = list(STORES_IDS.unique())

    for store_id in STORES_IDS:
        run = wandb.init(reinit=True)
        with run:
            config["data_loader"]["store_id"] = store_id
            logger.info("Getting Dataset..")
            start = time.time()
            X_train, y_train, X_test, y_test = gt_dataset(config)
            end = time.time() - start
            logger.info(f"Done! (elaped:{end} sec!)")

            logger.info(f"Start Training")
            seed_everything(SEED)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                eval_metric="rmse",
                early_stopping_rounds=5,
                callbacks = [wandb_callback]
            )
            

        model_name = "lgb_model_" + store_id + "_v" + str(1) + ".bin"
        pickle.dump(model, open(model_name, "wb"))
        # pickle.dump(estimator, open(model_name, "wb"))

        
        del X_train, y_train, X_test, y_test, model
        gc.collect()

        # # "Keep" models features for predictions
        # MODEL_FEATURES = features_columns


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="./config.json",
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--dstpath",
        default="./",
        type=str,
        help="destination path for saving model binaries (default: workspaceFolder)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="arch;args;lr"),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
