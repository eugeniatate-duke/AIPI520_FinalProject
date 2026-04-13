# ZIP-Level Electricity Demand Forecasting

## Overview
This project trains and compares machine learning models to predict monthly electricity consumption at the ZIP-code level using publicly available PG&E data.

Each data point represents electricity usage for a specific ZIP code and month. The goal is to learn patterns from historical usage and generate predictions for future consumption.

---

## Problem Statement
Can we accurately predict monthly electricity consumption for ZIP-level observations using historical data and engineered features?

---

## Dataset
- Source: PG&E Public Data  
  https://pge-energydatarequest.com/public_datasets  

- Data includes:
  - ZIP code
  - Month and year
  - Total electricity usage (kWh)
  - Number of customers
  - Customer class (Residential, Commercial)

The dataset was aggregated across multiple quarterly files into a single multi-year dataset.

---

## Target Variable
- `TOTALKWH`: Monthly total electricity consumption per ZIP code

---

## Data Processing
Steps performed:
- Merged multiple CSV files into one dataset
- Filtered to residential customers only
- Cleaned numeric columns 
- Removed invalid or missing values
- Sorted data by ZIP code and time

---

## Feature Engineering
The following features were created:

- `prev_m_usage` → previous month electricity usage  
- `rolling_3_avg` → average usage over last 3 months  
- `kwh_per_customer` → usage per customer  

These features help capture usage trends.

---

## Models

### 1. Linear Regression
- Simple and fast model
- Provides a baseline for comparison

### 2. Ridge Regression 
- Tested with different `alpha` values
- Did not significantly improve performance compared to LinReg

### 3. Neural Network (MLP)
- Built using PyTorch
- Architecture:
  - Input -> Dense(16) ->  ReLU  
  - Dense(8) -> ReLU  
  - Output layer  
- Trained from scratch
- Required scaling of features and target

---

## Model Optimization

### Linear / Ridge:
- Feature engineering
- Tested multiple alpha values for Ridge

### Neural Network:
- Tested different:
  - learning rates
  - number of epochs
- Discovered that scaling the target variable significantly improved performance

---

## Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

---

## Results

| Model | MAE | RMSE |
|------|------|------|
| Linear Regression | ~374k | ~720k |
| Neural Network | ~380k–420k | ~650k–720k |

### Key Findings:
- Linear Regression performed well as a baseline
- Neural Network slightly improved RMSE in some cases
- Neural Network required more preprocessing and tuning
- Ridge Regression did not significantly improve results

---

##  Observations
- Linear models struggled with peak electricity usage
- Neural networks handled nonlinear patterns better
- Scaling the target variable was critical for neural network performance
- Some models produced unrealistic negative predictions (limitation)

---

## Prediction System

A prediction script (`predict.py`) allows users to input (via hardcoding for now, not as actual input prompt):

- ZIP code  
- Month  
- Year  

The system:
1. Retrieves historical data for the ZIP code  
2. Automatically computes required features  
3. Generates predictions using:
   - Linear Regression  
   - Neural Network  

Example output:
{
‘zipcode’: 95212,
‘month’: 4,
‘year’: 2026,
‘linear_regression_prediction’: 4986231,
‘neural_network_prediction’: 4287750
}

---

## Project Structure

```
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── prepdata.py
│   ├── feature_eng.py
│   ├── train.py
│   ├── train_NN.py
│   └── predict.py
├── models/
├── README.md
``` 

---

##  How to Run

### 1. Data Preparation
```bash
python src/prepdata.py
```

### 2. Feature Engineering 
```
python src/feature_eng.py
``` 

### 3. Train Models 
```
python src/train.py
python src/train_nn.py 
```

### 4. Run Prediction
```
python src/predict.py
```

## Limitations: 

	- Model depends on availability of recent historical data
	- Does not explicitly use ZIP code as a feature (implicitly captured through history)
	- Limited long-term forecasting capability 
	- Neural network results vary slightly due to random initialization

## Future Improvements
	- Add explicit ZIP code encoding
    - Allow user prompt 
	- Improve neural network architecture or model selection 
	- Deploy as an API or web application

## Author
Eugenia Tate for AIPI 520 Final Project at Duke University 

