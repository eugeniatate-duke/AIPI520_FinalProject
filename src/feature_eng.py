import pandas as pd

def create_features(df):
    # sort data 
    df = df.sort_values(by=["ZIPCODE", "DATE"])

    # create previous month usage feature
    df["prev_m_usage"] = df.groupby("ZIPCODE")["TOTALKWH"].shift(1)

    # create rolling average for last 3 months
    df["rolling_3_avg"] = df.groupby("ZIPCODE")["TOTALKWH"].rolling(3).mean().reset_index(0, drop=True)

    # create average kwh per customer
    df["kwh_per_customer"] = df["TOTALKWH"] / df["TOTALCUSTOMERS"]

    # drop rows with missing values (from prev_m_avg feature)
    df = df.dropna()

    return df


if __name__ == "__main__":
    # load cleaned data
    df = pd.read_csv("data/processed/clean_data.csv")

    # create features
    df = create_features(df)

    # save new dataset in a new file 
    df.to_csv("data/processed/features_data.csv", index=False)

    print("features created and saved!")