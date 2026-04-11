import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
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

# create RF model
# model = RandomForestRegressor(
#     n_estimators=50,  # small to keep fast
#     max_depth=10,
#     random_state=42
# )

# create LinReg model
# model = LinearRegression()
# model = Ridge(alpha=1.0)

for alpha in [0.1, 1.0, 10.0]:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    print(f"alpha: {alpha}, rmse: {rmse}, mae: {mae}")
    # show predictions vs actual values
    results = pd.DataFrame({
        "actual": y_test.values,
        "predicted": preds
    })

    # show prediction error
    results["error"] = results["actual"] - results["predicted"]

    print(results.head(10))

# train model
# model.fit(X_train, y_train)

# get predictions
# preds = model.predict(X_test)

# evaluate MAE and RMSE 
# mae = mean_absolute_error(y_test, preds)
# rmse = np.sqrt(mean_squared_error(y_test, preds))

# print("mae:", mae)
# print("rmse:", rmse)

# show predictions vs actual values
# results = pd.DataFrame({
#     "actual": y_test.values,
#     "predicted": preds
# })

# # show prediction error
# results["error"] = results["actual"] - results["predicted"]

# print(results.head(10))