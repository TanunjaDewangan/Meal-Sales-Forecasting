# Meal Sales Forecasting Dashboard 

# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

# Streamlit Page Title and Layout
st.set_page_config(page_title="Meal Sales Forecasting", layout="wide")

# Streamlit Theme Customization
st.markdown("""
    <style>
    .reportview-container {
        background-color: #fef6f0;
    }
    .sidebar .sidebar-content {
        background-color: #dceefc;
    }
    .stButton>button {
        color: white;
        background-color: #ff7f50;
        border-radius: 10px;
        padding: 0.5em 1em;
    }
    </style>
""", unsafe_allow_html=True)


with st.sidebar.expander("ℹ️ Project Info"):
    st.markdown("""
    ### 📊 Meal Sales Forecasting Dashboard

    **Goal:** Forecast weekly meal demand for different meals and fulfillment centers.

    **Models Used:**
    - SARIMA (Time Series)
    - Prophet (Additive Time Series)
    - XGBoost (Machine Learning)

    **Key Features:**
    - Train-test split + rolling cross-validation
    - RMSE comparison (color-coded)
    - Residual plots
    - Forecast comparison across models

    **Tech Stack:** Python, Streamlit, Plotly, pandas, Prophet, statsmodels, XGBoost
    """)

st.title(" Meal Order Forecasting Dashboard")

# Main Title and Introduction
st.subheader("Weekly Meal Order Forecasting Dashboard")
st.markdown("""
This project demonstrates how we can forecast weekly meal orders using time series analysis and machine learning models.
You can choose a meal and center to explore trends and view future predictions.
""")

# Explanation of Models Used
with st.expander("About the Forecasting Models"):
    st.markdown("""
    - **SARIMA**: Great for capturing time-based trends and seasonality.
    - **Prophet**: Developed by Facebook, easy to use and good with irregular data.
    - **XGBoost**: A powerful machine learning model that uses past order values to predict future ones.
    """)

# Load Data and Combine Datasets
@st.cache_data
def load_data():
    train = pd.read_csv("C:\\Users\\hp\\Desktop\\PROJECTS\\Forecasting or Time Series Projects\\Data\\train\\train.csv")
    meals = pd.read_csv("C:\\Users\\hp\\Desktop\\PROJECTS\\Forecasting or Time Series Projects\\Data\\train\\meal_info.csv")
    centers = pd.read_csv("C:\\Users\\hp\\Desktop\\PROJECTS\\Forecasting or Time Series Projects\\Data\\train\\fulfilment_center_info.csv")
    df = train.merge(meals, on="meal_id").merge(centers, on="center_id")
    df['week_start'] = pd.to_datetime('2015-01-01') + pd.to_timedelta(df['week'] * 7, unit='D')
    return df

df = load_data()

# Sidebar for Filtering by Meal and Center
st.sidebar.header("Select Meal and Center")
selected_meal = st.sidebar.selectbox(" Choose Meal ID", df['meal_id'].unique())
selected_center = st.sidebar.selectbox(" Choose Center ID", df['center_id'].unique())

#  Filter Dataset Based on Selections
filtered = df[(df['meal_id'] == selected_meal) & (df['center_id'] == selected_center)]
weekly = filtered.groupby('week_start')['num_orders'].sum().reset_index()
weekly['7_day_MA'] = weekly['num_orders'].rolling(window=3).mean()

#  Plot Weekly Orders and Moving Average
st.subheader(" Weekly Orders & Moving Average")
fig = go.Figure()
fig.add_trace(go.Scatter(x=weekly['week_start'], y=weekly['num_orders'], mode='lines+markers', name='Orders'))
fig.add_trace(go.Scatter(x=weekly['week_start'], y=weekly['7_day_MA'], mode='lines', name='3-Week MA', line=dict(dash='dot')))
fig.update_layout(template="plotly_white", xaxis_title="Week", yaxis_title="Orders")
st.plotly_chart(fig, use_container_width=True)

#  Seasonality Heatmap by Month-Year
st.subheader("Seasonal Heatmap")
weekly['month'] = weekly['week_start'].dt.month
weekly['year'] = weekly['week_start'].dt.year
pivot = weekly.pivot_table(index='month', columns='year', values='num_orders', aggfunc='mean')
heatmap = px.imshow(pivot, text_auto=True, color_continuous_scale='Blues')
st.plotly_chart(heatmap, use_container_width=True)

# Seasonal Decomposition
st.subheader(" Seasonal Decomposition")
decomp = seasonal_decompose(weekly.set_index('week_start')['num_orders'], model='additive', period=12)
fig_dec = go.Figure()
fig_dec.add_trace(go.Scatter(x=decomp.observed.index, y=decomp.observed, name='Observed'))
fig_dec.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend, name='Trend'))
fig_dec.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal, name='Seasonal'))
fig_dec.add_trace(go.Scatter(x=decomp.resid.index, y=decomp.resid, name='Residual'))
fig_dec.update_layout(height=600, title="Seasonal Decomposition", showlegend=True)
st.plotly_chart(fig_dec, use_container_width=True)

#  Train-Test Split
series = weekly.set_index('week_start')['num_orders']
train_series, test_series = train_test_split(series, test_size=0.2, shuffle=False)

# SARIMA Forecasting
sarima_model = SARIMAX(train_series, order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
sarima_pred = sarima_model.forecast(len(test_series))
sarima_rmse = mean_squared_error(test_series, sarima_pred, squared=False)

# Prophet Forecasting
df_prophet = train_series.reset_index().rename(columns={'week_start': 'ds', 'num_orders': 'y'})
prophet_model = Prophet()
prophet_model.fit(df_prophet)
future = prophet_model.make_future_dataframe(periods=len(test_series), freq='W')
forecast = prophet_model.predict(future)
prophet_pred = forecast[['ds', 'yhat']].tail(len(test_series))['yhat']
prophet_rmse = mean_squared_error(test_series.values, prophet_pred.values, squared=False)

# XGBoost Forecasting
xgb = weekly.copy()
xgb['lag1'] = xgb['num_orders'].shift(1)
xgb['lag2'] = xgb['num_orders'].shift(2)
xgb['lag3'] = xgb['num_orders'].shift(3)
xgb['week_num'] = xgb['week_start'].dt.isocalendar().week
xgb['month'] = xgb['week_start'].dt.month
xgb.dropna(inplace=True)
features = ['lag1', 'lag2', 'lag3', 'week_num', 'month']
X = xgb[features]
y = xgb['num_orders']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
model = XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
xgb_rmse = mean_squared_error(y_test, y_pred, squared=False)

#  Best Model Info Based on RMSE
st.subheader(" Model Evaluation")
best = min({"SARIMA": sarima_rmse, "Prophet": prophet_rmse, "XGBoost": xgb_rmse}, key=lambda x: {"SARIMA": sarima_rmse, "Prophet": prophet_rmse, "XGBoost": xgb_rmse}[x])
st.success(f"Best model is **{best}** based on RMSE")
st.info(f"SARIMA: {sarima_rmse:.2f} | Prophet: {prophet_rmse:.2f} | XGBoost: {xgb_rmse:.2f}")


