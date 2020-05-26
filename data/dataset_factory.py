# coding: utf-8
import os, sys
import importlib
import numpy as np
import pandas as pd
import lightgbm as lgb

dataset_list = [
    "darker_magic",
]
sys.path.append("../")


def get_dataset(config):

    name = str(config["data_loader"]["type"])
    if name in dataset_list:
        X_train, y_train, X_test, y_test = importlib.import_module(
            "data." + name
        ).get_dataset(config)
    else:
        raise ("Invalid dataset name: {}".format(name))

    return X_train, y_train, X_test, y_test


def test():
    import time
    import argparse
    import sys

    sys.path.append(".")
    from parse_config import ConfigParser

    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="./config.json",
        type=str,
        help="config file path (default: None)",
    )

    # custom cli options to modify configuration from default values given in json file.
    config = ConfigParser.from_args(args)
    config["data_loader"]["store_id"] = "CA_1"

    print("Getting Dataset")
    print("Processing...")
    start = time.time()
    X_train, y_train, X_test, y_test = get_dataset(config)
    end = time.time() - start
    print(f"Done! (elaped:{end} sec!)")

    print(f"# of X_train:{len(X_train)}")
    print(f"# of y_train:{len(y_train)}")
    print(f"# of X_test:{len(X_test)}")
    print(f"# of y_test:{len(y_test)}")


if __name__ == "__main__":
    test()
