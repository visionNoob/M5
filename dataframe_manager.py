import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

TARGET = 'demand'
SEED = 1

def make_pca(
    df:pd.DataFrame,
    pca_col: str,
    n_days: int,
    keep_dim: int = 3
):
    print('PCA:', pca_col, n_days)
    pca_df = df[[pca_col, 'day_int', TARGET]]

    if pca_col != 'item_id':
        merge_base = pca_df[[pca_col,'day_int']]
        pca_df = pca_df.groupby([pca_col,'d_int'])[TARGET].agg(['sum']).reset_index()
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
    pca_df[res_cols] = pca_df[res_cols].fillna(0)
    
    # define PCA
    pca = PCA(random_state=SEED)
    pca.fit(pca_df[res_cols])
    pca_df[res_cols] = pca.transform(pca_df[res_cols])
    print("Explained variance ratio: ", pca.explained_variance_ratio_)
    
    keep_cols = res_cols[:keep_dim]
    print("Columns to keep: ", keep_cols)

    return pd.concat([df, pca_df[keep_cols]], axis=1)



def make_normal_lag(lag_day):
    NotImplemented
    

def merge_by_concat(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    merge_on: list
):
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1


def cutoff(df: pd.DataFrame, price_df: pd.DataFrame):
    '''
    release 안된 부분을 제거
    '''
    release_df = prices_df.groupby(['store_id','item_id'])['wm_yr_wk'].agg(['min']).reset_index()
    release_df.columns = ['store_id','item_id','release']
    df = merge_by_concat(df, release_df, ['store_id','item_id'])
    
    del release_df

    df = df[df['wm_yr_wk']>=df['release']]
    df = df.reset_index(drop=True)
    df['release'] = df['release'] - df['release'].min()
    df['release'] = df['release'].astype(np.int16)
    return df


def date_features(df: pd.DataFrame):

    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df.date.dt.day.astype(np.int8)
    df["month"] = df.date.dt.month.astype(np.int8)
    df['year'] = df.date.dt.year.astype(np.int8)
    df['year'] = (df['year'] - df['year'].min()).astype(np.int8)
    df["week_day"] = df.date.dt.weekday
    df["week_day"] = (df['week_day'] >= 5).astype(np.int8)
    df['week'] = df.date.dt.week

    df.drop(columns="date", inplace=True)

    return df

def price_features(
    prices_df: pd.DataFrame,
    calendar_df: pd.DataFrame
):
    '''
    For unit item and store_id
    '''
    prices_df['price_max'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('max')
    prices_df['price_min'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('min')
    prices_df['price_std'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('std')
    prices_df['price_mean'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('mean')
    prices_df['price_norm'] = prices_df['sell_price']/prices_df['price_max']
    
    calendar_prices = calendar_df[['wm_yr_wk','month','year']]
    calendar_prices = calendar_prices.drop_duplicates(subset=['wm_yr_wk'])
    prices_df = prices_df.merge(calendar_prices[['wm_yr_wk','month','year']], on=['wm_yr_wk'], how='left')
    del calendar_prices
    
    prices_df['price_momentum'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id'])['sell_price'].transform(lambda x: x.shift(1))
    prices_df['price_momentum_m'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','month'])['sell_price'].transform('mean')
    prices_df['price_momentum_y'] = prices_df['sell_price']/prices_df.groupby(['store_id','item_id','year'])['sell_price'].transform('mean')
    prices_df['price_nunique'] = prices_df.groupby(['store_id','item_id'])['sell_price'].transform('nunique')
    prices_df['item_nunique'] = prices_df.groupby(['store_id','sell_price'])['item_id'].transform('nunique')
    return prices_df

def sales_features(
    df: pd.DataFrame,
    mean_windows: list,
    std_windows: list,
):
    '''
    For unit item
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

