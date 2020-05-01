import pandas as pd
from sklearn.decomposition import PCA

TARGET = 'sales'
SEED = 1

def make_pca(
    df:pd.DataFrame,
    pca_col: str,
    n_days: int,
    keep_dim: int = 3
):
    print('PCA:', pca_col, n_days)
    pca_df = df[[pca_col, 'd', TARGET]]

    if pca_col != 'id':
        merge_base = pca_df[[pca_col,'d']]
        pca_df = pca_df.groupby([pca_col,'d'])[TARGET].agg(['sum']).reset_index()
        pca_df[TARGET] = pca_df['sum']
        del pca_df['sum']

    # min-max scaling
    pca_df[TARGET] = pca_df[TARGET]/pca_df[TARGET].max()
    LAG_DAYS = [col for col in range(1,n_days+1)]
    format_s = '{}_pca_'+pca_col+str(n_days)+'_{}'
    
    pca_df = pca_df.assign(**{
            format_s.format(col, l): pca_df.groupby([pca_col])[col].transform(lambda x: x.shift(l))
            for l in LAG_DAYS
            for col in [TARGET]
            })
    
    res_cols = list(pca_df)[3:]
    pca_df[pca_columns] = pca_df[pca_columns].fillna(0)
    
    # define PCA
    pca = PCA(random_state=SEED)
    pca.fit(pca_df[res_cols])
    pca_df[res_cols] = pca.transform(pca_df[res_cols])
    print("Explained variance ratio: ", pca.explained_variance_ratio)
    
    keep_cols = res_cols[:keep_dim]
    print("Columns to keep: ", keep_cols)

    return pca_df[keep_cols]



def make_normal_lag(lag_day):
    NotImplemented
    


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
