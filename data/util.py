import numpy as np
import pandas as pd
import os, sys, gc, time, warnings, pickle, psutil, random
from multiprocessing import Pool

warnings.filterwarnings("ignore")

## Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)


## Multiprocess Runs
def df_parallelize_run(func, t_split, N_CORES):
    num_cores = np.min([N_CORES, len(t_split)])
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, t_split), axis=1)
    pool.close()
    pool.join()
    return df


# Read data
def get_data_by_store(
    store_id,
    BASE,
    PRICE,
    CALENDAR,
    MEAN_ENC,
    LAGS,
    TARGET,
    mean_features,
    remove_features,
    START_TRAIN,
):

    # Read and contact basic feature
    df = pd.concat(
        [
            pd.read_pickle(BASE),
            pd.read_pickle(PRICE).iloc[:, 2:],
            pd.read_pickle(CALENDAR).iloc[:, 2:],
        ],
        axis=1,
    )

    # Leave only relevant store

    df = df[df["store_id"] == store_id]
    # With memory limits we have to read
    # lags and mean encoding features
    # separately and drop items that we don't need.
    # As our Features Grids are aligned
    # we can use index to keep only necessary rows
    # Alignment is good for us as concat uses less memory than merge.
    if MEAN_ENC:
        df2 = pd.read_pickle(MEAN_ENC)[mean_features]
        df2 = df2[df2.index.isin(df.index)]
        df = pd.concat([df, df2], axis=1)
        del df2  # to not reach memory limit
    
    if LAGS:
        df3 = pd.read_pickle(LAGS).iloc[:, 3:]
        df3 = df3[df3.index.isin(df.index)]
        df = pd.concat([df, df3], axis=1)
        del df3  # to not reach memory limit

    # Create features list
    features = [col for col in list(df) if col not in remove_features]
    df = df[["id", "d", TARGET] + features]

    # Skipping first n rows
    df = df[df["d"] >= START_TRAIN].reset_index(drop=True)

    return df, features


# Recombine Test set after training
def get_base_test(STORES_IDS):
    base_test = pd.DataFrame()

    for store_id in STORES_IDS:
        temp_df = pd.read_pickle("test_" + store_id + ".pkl")
        temp_df["store_id"] = store_id
        base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)

    return base_test


# Helper to make dynamic rolling lags
def make_lag(LAG_DAY, base_test, TARGET):
    lag_df = base_test[["id", "d", TARGET]]
    col_name = "sales_lag_" + str(LAG_DAY)
    lag_df[col_name] = (
        lag_df.groupby(["id"])[TARGET]
        .transform(lambda x: x.shift(LAG_DAY))
        .astype(np.float16)
    )
    return lag_df[[col_name]]


def make_lag_roll(LAG_DAY, base_test, TARGET):
    shift_day = LAG_DAY[0]
    roll_wind = LAG_DAY[1]
    lag_df = base_test[["id", "d", TARGET]]
    col_name = "rolling_mean_tmp_" + str(shift_day) + "_" + str(roll_wind)
    lag_df[col_name] = lag_df.groupby(["id"])[TARGET].transform(
        lambda x: x.shift(shift_day).rolling(roll_wind).mean()
    )
    return lag_df[[col_name]]
