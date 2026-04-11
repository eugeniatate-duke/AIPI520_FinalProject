import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# load data
df = pd.read_csv("data/processed/features_data.csv")

# select features 
features = [
    "MONTH",
    "YEAR",
    "TOTALCUSTOMERS",
    "prev_m_usage",
    "rolling_3_avg",
    "kwh_per_customer"
]

# target 
target = "TOTALKWH"

X = df[features]
y = df[target]

# split data 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# open results file
with open("results.txt", "w") as f:

    f.write("MODEL RESULTS\n")
    f.write("====================\n\n")
    # ---------------- LinRegression -----------
    f.write("linear regression\n")
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(X_train, y_train)
    lin_reg_preds = lin_reg_model.predict(X_test)

    # evaluate MAE and RMSE 
    lin_reg_mae = mean_absolute_error(y_test, lin_reg_preds)
    lin_reg_rmse = np.sqrt(mean_squared_error(y_test, lin_reg_preds))

    f.write(f"mae: {lin_reg_mae}\n")
    f.write(f"rmse: {lin_reg_rmse}\n")

    # save predictions
    lin_reg_results = pd.DataFrame({
        "actual": y_test.values,
        "predicted": lin_reg_preds
    })
    lin_reg_results["error"] = lin_reg_results["actual"] - lin_reg_results["predicted"]

    f.write("\nLinReg predictions:\n")
    f.write(lin_reg_results.head(10).to_string())
    f.write("\n\n")

    # --------------------- Ridge --------------------------------
    f.write("Ridge regression\n")
   
    for alpha in [0.1, 1.0, 10.0]:
        f.write(f"\nalpha: {alpha}\n")

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        f.write(f"mae: {mae}\n")
        f.write(f"rmse: {rmse}\n")
        
        # print(f"alpha: {alpha}, rmse: {rmse}, mae: {mae}")

        # show predictions vs actual values
        results = pd.DataFrame({
            "actual": y_test.values,
            "predicted": preds
        })

        # show prediction error
        results["error"] = results["actual"] - results["predicted"]

        # print(results.head(10))

        f.write("sample predictions:\n")
        f.write(results.head(10).to_string())
        f.write("\n")

print("results saved")