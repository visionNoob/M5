import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random
from data.util import seed_everything
from data.util import df_parallelize_run
from data.util import get_data_by_store
from data.util import get_base_test
from data.util import make_lag
from data.util import make_lag_roll


def get_dataset(config):
    store_id = config["data_loader"]["store_id"]
    BASE = os.path.join(config["data_loader"]["path_simple_fe"], "grid_part_1.pkl") if config["data_loader"]["path_simple_fe"] else None
    PRICE = os.path.join(config["data_loader"]["path_simple_fe"], "grid_part_2.pkl") if config["data_loader"]["path_simple_fe"] else None
    CALENDAR = os.path.join(config["data_loader"]["path_simple_fe"], "grid_part_3.pkl") if config["data_loader"]["path_simple_fe"] else None
    MEAN_ENC = os.path.join(
        config["data_loader"]["path_custom"], "mean_encoding_df.pkl"
    ) if config["data_loader"]["path_custom"] else None
    LAGS = os.path.join(config["data_loader"]["path_lags"], "lags_df_28.pkl") if config["data_loader"]["path_lags"] else None
    TARGET = config["data_loader"]["target"]
    mean_features = config["data_loader"]["mean_features"]
    remove_features = config["data_loader"]["remove_features"]
    remove_features.append(TARGET)
    START_TRAIN = config["data_loader"]["start_train"]
    END_TRAIN = config["data_loader"]["end_train"]
    P_HORIZON = config["data_loader"]["p_horizon"]

    # Get grid for current store
    grid_df, features_columns = get_data_by_store(
        store_id=store_id,
        BASE=BASE,
        PRICE=PRICE,
        CALENDAR=CALENDAR,
        MEAN_ENC=MEAN_ENC,
        LAGS=LAGS,
        TARGET=TARGET,
        mean_features=mean_features,
        remove_features=remove_features,
        START_TRAIN=START_TRAIN,
    )

    train_mask = grid_df["d"] <= END_TRAIN
    valid_mask = train_mask & (grid_df["d"] > (END_TRAIN - P_HORIZON))
    preds_mask = grid_df["d"] > (END_TRAIN - 100)

    X_train = grid_df[train_mask][features_columns]
    y_train = grid_df[train_mask][TARGET]
    X_test = grid_df[valid_mask][features_columns]
    y_test = grid_df[valid_mask][TARGET]

    grid_df = grid_df[preds_mask].reset_index(drop=True)
    keep_cols = [col for col in list(grid_df) if "_tmp_" not in col]
    grid_df = grid_df[keep_cols]
    grid_df.to_pickle("test_" + store_id + ".pkl")
    del grid_df

    # X_train.reset_index(drop=True, inplace=True)
    # y_train.reset_index(drop=True, inplace=True)
    # X_test.reset_index(drop=True, inplace=True)
    # y_test.reset_index(drop=True, inplace=True)

    # X_train = pd.DataFrame(X_train)
    # y_train = pd.DataFrame(y_train)
    # X_test = pd.DataFrame(X_test)
    # y_test = pd.DataFrame(y_test)

    return X_train, y_train, X_test, y_test
