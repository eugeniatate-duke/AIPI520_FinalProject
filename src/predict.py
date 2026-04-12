import pandas as pd
import joblib
import torch , torch.nn as nn


# load model
# model = joblib.load("models/lin_reg_model.pkl")

# load data
df = pd.read_csv("data/processed/features_data.csv")

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.model(x)

 # get the month before the target month
def get_previous_month(month, year):
    if month == 1:
        return 12, year - 1
    else:
        return month - 1, year

# build a list of the 3 months before the target month
def get_last_3_months(month, year):
    months = []

    for _ in range(3):
        month, year = get_previous_month(month, year)
        months.append((month, year))

    return months


def build_features_from_zip(zipcode, target_month, target_year):
    # filter data for one zip code
    zip_df = df[df["ZIPCODE"] == zipcode].copy()

    if zip_df.empty:
        raise ValueError("zip code not found")

    # get previous month
    prev_month, prev_year = get_previous_month(target_month, target_year)

    prev_row = zip_df[
        (zip_df["MONTH"] == prev_month) & (zip_df["YEAR"] == prev_year)
    ]

    if prev_row.empty:
        raise ValueError("previous month data not found for this zip code")

    prev_m_usage = prev_row.iloc[0]["TOTALKWH"]
    customers = prev_row.iloc[0]["TOTALCUSTOMERS"]
    kwh_per_customer = prev_row.iloc[0]["kwh_per_customer"]

    # get previous 3 months for rolling average
    last_3 = get_last_3_months(target_month, target_year)

    rolling_rows = []
    for m, y in last_3:
        row = zip_df[(zip_df["MONTH"] == m) & (zip_df["YEAR"] == y)]
        if not row.empty:
            rolling_rows.append(row.iloc[0]["TOTALKWH"])

    if len(rolling_rows) < 3:
        raise ValueError("not enough data to calculate rolling average")

    rolling_3_avg = sum(rolling_rows) / 3

    # create a small 1 row dataframe with model inputs
    input_df = pd.DataFrame([{
        "MONTH": target_month,
        "YEAR": target_year,
        "TOTALCUSTOMERS": customers,
        "prev_m_usage": prev_m_usage,
        "rolling_3_avg": rolling_3_avg,
        "kwh_per_customer": kwh_per_customer
    }])

    return input_df

def predict_with_linreg(input_df):
    # load saved linreg model
    lin_model = joblib.load("models/lin_reg_model.pkl")

    # make prediction
    pred = lin_model.predict(input_df)[0]

    return pred

def predict_with_nn(input_df):
    # load scalers
    x_scaler = joblib.load("models/x_scaler.pkl")
    y_scaler = joblib.load("models/y_scaler.pkl")

    # load saved NN
    nn_model = SimpleNN(input_size=6)
    nn_model.load_state_dict(torch.load("models/nn_model.pt"))
    nn_model.eval()

    # scale input
    input_scaled = x_scaler.transform(input_df)

    # convert to tensor
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # predict
    with torch.no_grad():
        pred_scaled = nn_model(input_tensor).numpy()

    # convert back to original kwh scale
    pred = y_scaler.inverse_transform(pred_scaled)[0][0]

    return pred

def predict_usage(zipcode, target_month, target_year):
    # build inference model features from zip data
    input_df = build_features_from_zip(zipcode, target_month, target_year)

    # make prediction
    # prediction = model.predict(input_df)[0]

     # get predictions from both models
    linreg_pred = predict_with_linreg(input_df)
    nn_pred = predict_with_nn(input_df)

    # return prediction
    return {
        "zipcode": zipcode,
        "month": target_month,
        "year": target_year,
        "linear_regression_prediction": linreg_pred,
        "neural_network_prediction": nn_pred
    }

# values are hardcoded but can be modified to accept user input 
if __name__ == "__main__":
    zipcode = 95212
    target_month = 4
    target_year = 2026

    pred = predict_usage(zipcode, target_month, target_year)
    print("predicted results:", pred)