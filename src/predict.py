import pandas as pd
import joblib 

# load trained model 
model = joblib.load("models/lin_reg_model.pkl")

def predict_usage(month, year, customers, prev_usage, rolling_avg, kwh_per_customer):
    # create dataframe with one row
    data = pd.DataFrame([{
        "MONTH": month,
        "YEAR": year,
        "TOTALCUSTOMERS": customers,
        "prev_m_usage": prev_usage,
        "rolling_3_avg": rolling_avg,
        "kwh_per_customer": kwh_per_customer
    }])

    prediction = model.predict(data)

    return prediction[0]


if __name__ == "__main__":
    pred = predict_usage(6, 2026, 300, 200000, 220000, 700)
    print("predicted kwh:", pred)