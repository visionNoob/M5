import os, sys
import pandas as pd


def get_base_test(STORES_IDS):
    base_test = pd.DataFrame()

    for store_id in STORES_IDS:
        temp_df = pd.read_pickle("test_" + store_id + ".pkl")
        temp_df["store_id"] = store_id
        base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)

    return base_test


ORIGINAL = "./input/m5-forecasting-accuracy/"


def predict():
    STORES_IDS = pd.read_csv(ORIGINAL + "sales_train_validation.csv")["store_id"]
    STORES_IDS = list(STORES_IDS.unique())
    # Create Dummy DataFrame to store predictions
    all_preds = pd.DataFrame()

    # Join back the Test dataset with
    # a small part of the training data
    # to make recursive features
    base_test = get_base_test(STORES_IDS)

    # Timer to measure predictions time
    main_time = time.time()

    # Loop over each prediction day
    # As rolling lags are the most timeconsuming
    # we will calculate it for whole day
    for PREDICT_DAY in range(1, 29):
        print("Predict | Day:", PREDICT_DAY)
        start_time = time.time()

        # Make temporary grid to calculate rolling lags
        grid_df = base_test.copy()
        grid_df = pd.concat(
            [grid_df, df_parallelize_run(make_lag_roll, ROLS_SPLIT)], axis=1
        )

        for store_id in STORES_IDS:

            # Read all our models and make predictions
            # for each day/store pairs
            model_path = "lgb_model_" + store_id + "_v" + str(VER) + ".bin"
            if USE_AUX:
                model_path = AUX_MODELS + model_path

            estimator = pickle.load(open(model_path, "rb"))

            day_mask = base_test["d"] == (END_TRAIN + PREDICT_DAY)
            store_mask = base_test["store_id"] == store_id

            mask = (day_mask) & (store_mask)
            base_test[TARGET][mask] = estimator.predict(grid_df[mask][MODEL_FEATURES])

        # Make good column naming and add
        # to all_preds DataFrame
        temp_df = base_test[day_mask][["id", TARGET]]
        temp_df.columns = ["id", "F" + str(PREDICT_DAY)]
        if "id" in list(all_preds):
            all_preds = all_preds.merge(temp_df, on=["id"], how="left")
        else:
            all_preds = temp_df.copy()

        print(
            "#" * 10,
            " %0.2f min round |" % ((time.time() - start_time) / 60),
            " %0.2f min total |" % ((time.time() - main_time) / 60),
            " %0.2f day sales |" % (temp_df["F" + str(PREDICT_DAY)].sum()),
        )
        del temp_df

    all_preds = all_preds.reset_index(drop=True)
    all_preds


if __name__ == "__main__":
    predict()
