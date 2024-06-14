# Weather-Forecasting-Monitoring-System-for-Sri-Lanka

## Overview
The Sri Lanka Weather Forecasting Monitoring System is a comprehensive project aimed at accurately predicting rainfall in various regions of Sri Lanka. This project leverages advanced data science techniques, including Time Series Analysis (SARIMA), Linear Regression, and Support Vector Machine Regression, to develop a robust and reliable model for weather prediction. The primary goal is to enhance the accuracy and reliability of weather forecasts to aid in resource management, agriculture, and disaster preparedness.|

## Features
- **Multiple Forecasting Models:** Implements SARIMA, Linear Regression, and Support Vector Machine Regression to predict rainfall.
- **Visualization Tools:** Utilizes Microsoft Power BI for creating insightful visualizations.
- **Error Analysis:** Uses Mean Squared Error (MSE) for model evaluation and selection.
- **Future Predictions:** Provides rainfall predictions for the coming years based on the best-performing model.
- **Comprehensive Data Handling:** Processes and analyzes large datasets efficiently using Python Jupyter Notebook.

## Data Science Techniques

### Time Series Analysis (SARIMA)
The Seasonal Autoregressive Integrated Moving Average (SARIMA) model is a pivotal component of our rainfall prediction system, effectively capturing the seasonality and trends inherent in the historical rainfall data of Sri Lanka. This model is particularly well-suited for time series data exhibiting regular seasonal patterns, making it an ideal choice for weather forecasting. Here's an in-depth look at how SARIMA is utilized in this project:

#### Data Stationarity
To apply SARIMA, the time series data must be stationary, meaning its statistical properties (mean, variance, and autocorrelation) are constant over time. Achieving stationarity involves:

- **Trend Removal:** Differencing the data (subtracting the previous observation from the current observation) to eliminate trends. This is known as the 'd' parameter in ARIMA.
- **Seasonality Adjustment:** Applying seasonal differencing to remove seasonality. This is the 'D' parameter in SARIMA, representing the number of seasonal differences.
- **Transformation:** Applying logarithmic or other transformations if needed to stabilize the variance.

In the project, these steps are implemented using Python's statsmodels library, which provides tools for differencing and testing stationarity (e.g., Augmented Dickey-Fuller test).

#### Model Identification
Identifying the correct order of the ARIMA and seasonal components (SARIMA) is crucial for accurate forecasting. This involves:

- **Autocorrelation Function (ACF):** Analyzing the ACF plot to identify the number of lags (q) in the Moving Average (MA) component.
- **Partial Autocorrelation Function (PACF):** Examining the PACF plot to determine the number of lags (p) in the Autoregressive (AR) component.
- **Seasonal Parameters:** Determining the seasonal order (P, D, Q, S) based on the seasonality observed in the data. Here, S represents the seasonal period (e.g., 12 for monthly data with yearly seasonality).

For this project, the ACF and PACF plots are generated and interpreted using Python's matplotlib and statsmodels libraries, helping to identify the appropriate model parameters.

#### Parameter Estimation
Once the model order is identified, the next step is to estimate the parameters. This is achieved through:

- **Maximum Likelihood Estimation (MLE):** A statistical method used to estimate the parameters of the SARIMA model by maximizing the likelihood function. MLE ensures that the estimated parameters are those that make the observed data most probable.

In the project, the SARIMAX function from the statsmodels library is utilized for parameter estimation, which provides built-in capabilities for fitting SARIMA models using MLE.

#### Model Diagnostics
After fitting the SARIMA model, it is essential to validate its adequacy through diagnostic checks:

- **Residual Analysis:** Checking the residuals (the differences between the observed and fitted values) to ensure they behave like white noise (i.e., no autocorrelation, constant mean, and variance).
- **Ljung-Box Test:** A statistical test applied to the residuals to check for the presence of autocorrelation. If the residuals pass this test, it indicates a good model fit.
- **Normality Test:** Ensuring the residuals are normally distributed, often using Q-Q plots and other statistical tests.

In this project, residual diagnostics are conducted using Python's statsmodels and scipy libraries, ensuring the fitted SARIMA model is reliable and accurate.

#### Implementation Steps in the Project
- **Data Preprocessing:** The historical rainfall data is first preprocessed to handle missing values and outliers. The data is then differenced to achieve stationarity.
- **Model Selection:** ACF and PACF plots are generated to determine the initial values of p, d, q, P, D, Q, and S.
- **Parameter Estimation:** The SARIMAX function from statsmodels is used to fit the SARIMA model and estimate the parameters using MLE.
- **Model Fitting:** The SARIMA model is trained on the historical data.
- **Diagnostics:** Residuals are analyzed using statistical tests and plots to validate the model fit.
- **Forecasting:** The validated SARIMA model is used to predict future rainfall patterns, providing insights for the coming years.

By meticulously following these steps, the project ensures that the SARIMA model is both accurate and reliable, making it a powerful tool for rainfall forecasting in Sri Lanka.
