import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import torch
import wandb
from tqdm import tqdm


def train_model(
    model, train_loader, criterion, optimizer, epoch, scheduler=None, history=None,
):
    model.train()
    total_loss = 0

    t = tqdm(train_loader)

    for i, d in enumerate(t):
        item = d["features"].cuda().float()
        y_batch = d["label"].cuda().float()
        optimizer.zero_grad()
        out = model(item)
        loss = criterion(out, y_batch)
        total_loss += loss
        t.set_description(
            f"Epoch {epoch+1} : , LR: %6f, Loss: %.4f"
            % (optimizer.state_dict()["param_groups"][0]["lr"], total_loss / (i + 1)),
        )
        if history is not None:
            history.loc[epoch + i / len(train_loader), "train_loss"] = loss.data.cpu().numpy()
            history.loc[epoch + i / len(train_loader), "lr"] = optimizer.state_dict()["param_groups"][0]["lr"]
        loss.backward()
        optimizer.step()

    return loss.data.cpu().numpy() / len(train_loader)


def evaluate_model(model, val_loader, criterion, epoch, scheduler=None, history=None):
    model.eval()
    loss = 0
    pred_list = []
    real_list = []
    RMSE_list = []
    with torch.no_grad():
        for i, d in enumerate(tqdm(val_loader)):
            item = d["features"].cuda().float()
            y_batch = d["label"].cuda().float()

            o1 = model(item)
            l1 = criterion(o1, y_batch)
            loss += l1

            o1 = o1.cpu().numpy()
            y_batch = y_batch.cpu().numpy()

            for pred, real in zip(o1, y_batch):
                rmse = np.sqrt(metrics.mean_squared_error(real, pred))
                RMSE_list.append(rmse)
                pred_list.append(pred)
                real_list.append(real)

    loss /= len(val_loader)
    if scheduler is not None:
        scheduler.step(loss)
    print(f"\n Dev loss: %.4f RMSE : %.4f" % (loss, np.mean(RMSE_list)))

    return loss


def reduce_mem_usage(df: pd.DataFrame, verbose=True):
    """
    For efficient memory usage, set the df data type.

    Args:
        df: pd.DataFrame

    Returns
        df: pd.DataFrame

    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem,
            ),
        )
    return df


def read_data(PATH: str):
    """
    Args:
        PATH: str

    Return
        List of DataFrames, [calendar, sell_prices, sales_train_validation, submission]
    """
    print("Reading files...")
    calendar = pd.read_csv(f"{PATH}/calendar.csv")
    calendar = reduce_mem_usage(calendar)
    print(
        "Calendar has {} rows and {} columns".format(
            calendar.shape[0], calendar.shape[1],
        ),
    )
    sell_prices = pd.read_csv(f"{PATH}/sell_prices.csv")
    sell_prices = reduce_mem_usage(sell_prices)
    print(
        "Sell prices has {} rows and {} columns".format(
            sell_prices.shape[0], sell_prices.shape[1],
        ),
    )
    sales_train_validation = pd.read_csv(f"{PATH}/sales_train_validation.csv")
    print(
        "Sales train validation has {} rows and {} columns".format(
            sales_train_validation.shape[0], sales_train_validation.shape[1],
        ),
    )
    submission = pd.read_csv(f"{PATH}/sample_submission.csv")
    return calendar, sell_prices, sales_train_validation, submission
