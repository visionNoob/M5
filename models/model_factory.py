# coding: utf-8
import os
import numpy as np
import pandas as pd
import lightgbm as lgb


def get_network(config, mode="train"):

    name = str(config["arch"]["type"])

    if name in ["lightgbm"]:
        model = _get_model_instance(name)
        args = dict(config["arch"]["args"])
        model = model(**args)

    else:
        raise ("Invalid model name: {}".format(name))

    return model


def _get_model_instance(name):
    return {"lightgbm": lgb.LGBMRegressor}[name]
