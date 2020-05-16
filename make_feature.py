

import argparse
from dataframe_manager import (
    make_pca, merge_by_concat, cutoff, date_features, price_features, sales_features
    )

import pandas as pd
import numpy as np
from utils import reduce_mem_usage, sizeof_fmt


index_columns = ['id','item_id','dept_id','cat_id','store_id','state_id']

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--sales_dataset', type=str, default="./data/m5-forecasting-accuracy/sales_train_validation.csv", 
        help="sales dataset folder path")
    
    parser.add_argument(
        '--prices_dataset', type=str, default="./data/m5-forecasting-accuracy/sell_prices.csv", 
        help="prices dataset folder path")
    parser.add_argument(
        '--calendar_dataset', type=str, default="./data/m5-forecasting-accuracy/calendar.csv", 
        help="prices dataset folder path")

    parser.add_argument('--target', type=str, default='sales', help="")
    parser.add_argument('--end_train', type=int, default=1913, help="")
    parser.add_argument('--grid_df1', type=str, default='./grid_df/grid_part_1.pkl', help="")
    parser.add_argument('--grid_df2', type=str, default='./grid_df/grid_part_2.pkl', help="")
    parser.add_argument('--grid_df3', type=str, default='./grid_df/grid_part_3.pkl', help="")
    parser.add_argument('--main_index', type=str, default='id,d', help="")
    parser.add_argument('--state_id', type=str, default='CA', help="")
    parser = parser.parse_args()
    return parser


def make_grid_df(train_df, config):
    grid_df = pd.melt(train_df, 
                    id_vars = index_columns, 
                    var_name = 'd', 
                    value_name = config.target)
    return grid_df

def make_add_grid_df(train_df):
    add_grid = pd.DataFrame()
    for i in range(1,29):
        temp_df = train_df[index_columns]
        temp_df = temp_df.drop_duplicates()
        temp_df['d'] = 'd_'+ str(config.end_train + i)
        temp_df[config.target] = np.nan
        add_grid = pd.concat([add_grid,temp_df])
    return add_grid


def make_release_df(prices_df):
    release_df = prices_df.groupby(['store_id','item_id'])['wm_yr_wk'].agg(['min']).reset_index()
    release_df.columns = ['store_id','item_id','release']
    return release_df


if __name__ == '__main__':
    
    config = get_config()
    train_df = pd.read_csv(config.sales_dataset)
    prices_df = pd.read_csv(config.prices_dataset)
    calendar_df = pd.read_csv(config.calendar_dataset)

    print('Create Grid')
    grid_df = make_grid_df(train_df, config)
    print('Train rows:', len(train_df), len(grid_df))
    print("{:>20}: {:>8}".format('Original grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

    add_grid = make_add_grid_df(train_df)
    grid_df = pd.concat([grid_df,add_grid])
    grid_df = grid_df.reset_index(drop=True)
    
    for col in index_columns:
        grid_df[col] = grid_df[col].astype('category')

    # Let's check again memory usage
    print("{:>20}: {:>8}".format('Reduced grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

    print('Release week')
    grid_df = merge_by_concat(grid_df, calendar_df[['wm_yr_wk','d']], ['d'])
    grid_df = cutoff(grid_df, prices_df)
    # Let's check again memory usage
    print("{:>20}: {:>8}".format('Reduced grid_df',sizeof_fmt(grid_df.memory_usage(index=True).sum())))

    print('Save Part 1')
    grid_df.to_pickle(config.grid_df1)
    print('Size:', grid_df.shape)


    prices_df = price_features(prices_df, calendar_df)
    # Merge Prices
    original_columns = list(grid_df)
    grid_df = grid_df.merge(prices_df, on=['store_id','item_id','wm_yr_wk'], how='left')
    keep_columns = [col for col in list(grid_df) if col not in original_columns]
    
    main_index = config.main_index.split(",")
    grid_df = grid_df[main_index + keep_columns]

    grid_df = reduce_mem_usage(grid_df)
    grid_df.to_pickle(config.grid_df2)
    

    grid_df = pd.read_pickle(config.grid_df1)
    grid_df = grid_df[main_index]

    icols = ['date',
            'd',
            'event_name_1',
            'event_type_1',
            'event_name_2',
            'event_type_2',
            'snap_CA',
            'snap_TX',
            'snap_WI']

    grid_df = grid_df.merge(calendar_df[icols], on=['d'], how='left')

    icols = ['event_name_1',
            'event_type_1',
            'event_name_2',
            'event_type_2',
            'snap_CA',
            'snap_TX',
            'snap_WI']

    for col in icols:
        grid_df[col] = grid_df[col].astype('category')

    # Convert to DateTime
    grid_df = date_features(grid_df)
    # Remove date
    print('Save part 3')

    # Safe part 3
    grid_df.to_pickle(config.grid_df3)
    print('Size:', grid_df.shape)


    # Convert 'd' to int
    grid_df = pd.read_pickle(config.grid_df1)
    grid_df['d'] = grid_df['d'].apply(lambda x: x[2:]).astype(np.int16)

    # Remove 'wm_yr_wk'
    # as test values are not in train set
    del grid_df['wm_yr_wk']
    grid_df.to_pickle(config.grid_df1)

    
