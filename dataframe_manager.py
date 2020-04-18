import pandas as pd


def date_features(df: pd.DataFrame):

    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df.date.dt.day
    df["month"] = df.date.dt.month
    df["week_day"] = df.date.dt.weekday
    df.drop(columns="date", inplace=True)

    return df


def sales_features(
    df: pd.DataFrame,
    mean_windows: list,
    std_windows: list,
):
    '''
    Args:
        df: pd.DataFrame
        mean_windows: List of rolling mean window_size, list
        std_window: List of rolling std window_size, list
    Returns
        df: pd.DataFrame
    '''
    for window in mean_windows:
        df['rolling_price_mean_t' + str(window)] = df["sell_price"].transform(lambda x: x.rolling(window).mean())

    for window in std_windows:
        df['rolling_price_std_t' + str(window)] = df['sell_price'].transform(lambda x: x.rolling(window).std())
    df.sell_price.fillna(0, inplace=True)
    
        df['rolling_mean_t' + str(window)] = df["demand"].transform(lambda x: x.rolling(window).mean())

    for window in std_windows:
        df['rolling_std_t' + str(window)] = df['demand'].transform(lambda x: x.rolling(window).std())
    df.sell_price.fillna(0, inplace=True)
    return df


def demand_features(
    df: pd.DataFrame,
    shift: int,
    mean_windows: list,
    std_windows: list,
):
    '''
    Args:
        df: pd.DataFrame
        shift: data shifting to prevent data leakage.
        mean_windows: List of rolling mean window_size, list
        std_window: List of rolling std window_size, list
    Returns
        df: pd.DataFrame
    '''
    df["lag_t" + str(shift)] = df["demand"].transform(lambda x: x.shift(shift))

    for window in mean_windows:
        df['rolling_mean_t' + str(window)] = df["demand"].transform(lambda x: x.shift(shift).rolling(window).mean())

    for window in std_windows:
        df['rolling_std_t' + str(window)] = df['demand'].transform(lambda x: x.shift(shift).rolling(window).std())

    df.fillna(0, inplace=True)
    return df
