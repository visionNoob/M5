
import pandas as pd
import joblib
import torch


class DataLoading:
    def __init__(self, df, label, train_window = 28, predicting_window=28):
        self.df = df.values
        self.label = label
        self.train_window = train_window
        self.predicting_window = predicting_window

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, item):
        df_item = self.df[item]
        item_id = df_item[0]
        day_int = df_item[1]
        
        item_npy = joblib.load(f"something_spl/{item_id}.npy")
        item_npy_demand = item_npy[:,0]
        features = item_npy[day_int-self.train_window:day_int]
        predicted_demand = item_npy_demand[day_int:day_int+self.predicting_window]

        item_label = self.label.transform([item_id])
        item_onehot = [0] * 3049
        item_onehot[item_label[0]] = 1

        list_features = []
        for f in features:
            one_f = []
            one_f.extend(item_onehot)
            one_f.extend(f)
            list_features.append(one_f)

        return {
            "features" : torch.Tensor(list_features),
            "label" : torch.Tensor(predicted_demand)
        }