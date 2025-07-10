## Meal-Sales-Forecasting
## Meal Sales Forecasting Project - Streamlit Dashboard

This project is an end-to-end time series forecasting application built with Streamlit that predicts weekly
meal orders. It compares the performance of three forecasting models: SARIMA, Facebook Prophet, and
XGBoost.

## Business Problem:

A food delivery company needs accurate weekly demand forecasts to optimize inventory, reduce waste,
and ensure smooth supply chain operations across its fulfillment centers.

## Objective:

Minimize: Forecasting error (e.g., RMSE)

Maximize: Forecast accuracy and business interpretability

## Constraints:

Minimum: Historical data availability per meal-center combination

Maximum: Forecast horizon limited to 10 weeks

## Project Structure:

meal_sales_forecasting/

├── train.csv
├── meal_info.csv
├── fulfilment_center_info.csv
├── meal_forecasting_app.py # Streamlit app file
├── README.md # Project overview (this file)
└── requirements.txt # Python dependencies

## Features:

1.Filterable by Meal ID and Center ID

2.Weekly order trend with moving average

3.Seasonality heatmap (month vs year)

4.Model comparison: SARIMA, Prophet, XGBoost

5.RMSE evaluation with best model highlight

## How to Run Locally:

1. Clone the Repository
git clone https://github.com/your-username/meal-sales-forecasting.git
cd meal-sales-forecasting

2. Install Dependencies
pip install -r requirements.txt

3. Launch Streamlit App
streamlit run meal_forecasting_app.py
The app will open in your browser at http\://localhost:8501

## Models Used & Model Description:

SARIMA Classical time series model capturing seasonality

Prophet Business-friendly time series tool from Facebook

XGBoost ML model trained on lag features
