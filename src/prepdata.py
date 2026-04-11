import pandas as pd
import glob
import os

# path to raw data
DATA_PATH = "data/raw/*.csv"

def load_and_merge_data():
    files = glob.glob(DATA_PATH)
    
    df_list = []
    
    for file in files:
        print(f"Loading {file}")
        df = pd.read_csv(file)
        df_list.append(df)
    
    combined_df = pd.concat(df_list, ignore_index=True)
    
    return combined_df

def clean_data(df):
    # make all column names uppercase and clean
    df.columns = [col.strip().upper() for col in df.columns]

    # convert totalkwh to numeric 
    df["TOTALKWH"] = df["TOTALKWH"].astype(str).str.replace(",", "")
    df["TOTALKWH"] = pd.to_numeric(df["TOTALKWH"], errors="coerce")

    # convert totalcustomers to numeric
    df["TOTALCUSTOMERS"] = df["TOTALCUSTOMERS"].astype(str).str.replace(",", "")
    df["TOTALCUSTOMERS"] = pd.to_numeric(df["TOTALCUSTOMERS"], errors="coerce")

    # filter residential only
    df = df[df["CUSTOMERCLASS"].str.contains("Residential", case=False, na=False)]

    # remove rows where totalkwh is missing or zero
    df = df[df["TOTALKWH"].notna()]
    df = df[df["TOTALKWH"] > 0]

    # create a date column
    df["DATE"] = pd.to_datetime(dict(year=df["YEAR"], month=df["MONTH"], day=1))

    # sort values
    df = df.sort_values(by=["ZIPCODE", "DATE"])

    return df

def save_processed(df):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/clean_data.csv", index=False)
    print("Saved cleaned data!")

if __name__ == "__main__":
    df = load_and_merge_data()
    print("Merged shape:", df.shape)
    
    df = clean_data(df)
    print("Cleaned shape:", df.shape)
    
    save_processed(df)