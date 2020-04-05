
import pandas as pd


def date_features(df: pd.DataFrame):
    
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df.date.dt.day
    df["month"] = df.date.dt.month
    df["week_day"] = df.date.dt.weekday
    df.drop(columns="date", inplace=True)

    return df

def sales_features(df: pd.DataFrame):
    df.sell_price.fillna(0, inplace=True)

    return df

def demand_features(df: pd.DataFrame):
    df["lag_t28"] = df["demand"].transform(lambda x: x.shift(28))
    df["rolling_mean_t7"] = df["demand"].transform(lambda x:x.shift(28).rolling(7).mean())
    df['rolling_mean_t30'] = df['demand'].transform(lambda x: x.shift(28).rolling(30).mean())
    df['rolling_mean_t60'] = df['demand'].transform(lambda x: x.shift(28).rolling(60).mean())
    df['rolling_mean_t90'] = df['demand'].transform(lambda x: x.shift(28).rolling(90).mean())
    df['rolling_mean_t180'] = df['demand'].transform(lambda x: x.shift(28).rolling(180).mean())
    df['rolling_std_t7'] = df['demand'].transform(lambda x: x.shift(28).rolling(7).std())
    df['rolling_std_t30'] = df['demand'].transform(lambda x: x.shift(28).rolling(30).std())
    df.fillna(0, inplace=True)

    return df