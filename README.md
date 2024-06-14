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

In the project, these steps are implemented using Python statsmodels library, which provides tools for differencing and testing stationarity (e.g., Augmented Dickey-Fuller test).

#### Model Identification
Identifying the correct order of the ARIMA and seasonal components (SARIMA) is crucial for accurate forecasting. This involves:

- **Autocorrelation Function (ACF):** Analyzing the ACF plot to identify the number of lags (q) in the Moving Average (MA) component.
- **Partial Autocorrelation Function (PACF):** Examining the PACF plot to determine the number of lags (p) in the Autoregressive (AR) component.
- **Seasonal Parameters:** Determining the seasonal order (P, D, Q, S) based on the seasonality observed in the data. Here, S represents the seasonal period (e.g., 12 for monthly data with yearly seasonality).

For this project, the ACF and PACF plots are generated and interpreted using Python matplotlib and statsmodels libraries, helping to identify the appropriate model parameters.

#### Parameter Estimation
Once the model order is identified, the next step is to estimate the parameters. This is achieved through:

- **Maximum Likelihood Estimation (MLE):** A statistical method used to estimate the parameters of the SARIMA model by maximizing the likelihood function. MLE ensures that the estimated parameters are those that make the observed data most probable.

In the project, the SARIMAX function from the statsmodels library is utilized for parameter estimation, which provides built-in capabilities for fitting SARIMA models using MLE.

#### Model Diagnostics
After fitting the SARIMA model, it is essential to validate its adequacy through diagnostic checks:

- **Residual Analysis:** Checking the residuals (the differences between the observed and fitted values) to ensure they behave like white noise (i.e., no autocorrelation, constant mean, and variance).
- **Ljung-Box Test:** A statistical test applied to the residuals to check for the presence of autocorrelation. If the residuals pass this test, it indicates a good model fit.
- **Normality Test:** Ensuring the residuals are normally distributed, often using Q-Q plots and other statistical tests.

In this project, residual diagnostics are conducted using Python statsmodels and scipy libraries, ensuring the fitted SARIMA model is reliable and accurate.

#### Implementation Steps in the Project
- **Data Preprocessing:** The historical rainfall data is first preprocessed to handle missing values and outliers. The data is then differenced to achieve stationarity.
- **Model Selection:** ACF and PACF plots are generated to determine the initial values of p, d, q, P, D, Q, and S.
- **Parameter Estimation:** The SARIMAX function from statsmodels is used to fit the SARIMA model and estimate the parameters using MLE.
- **Model Fitting:** The SARIMA model is trained on the historical data.
- **Diagnostics:** Residuals are analyzed using statistical tests and plots to validate the model fit.
- **Forecasting:** The validated SARIMA model is used to predict future rainfall patterns, providing insights for the coming years.

By meticulously following these steps, the project ensures that the SARIMA model is both accurate and reliable, making it a powerful tool for rainfall forecasting in Sri Lanka.

### Linear Regression
Linear Regression is a fundamental statistical technique used in this project to establish a relationship between rainfall and various predictors. This method helps in understanding and quantifying how changes in the predictors affect the amount of rainfall. Here's a detailed breakdown of how Linear Regression is applied in the project:

#### Model Specification
The first step in Linear Regression is specifying the model, which involves defining the dependent and independent variables:

- **Dependent Variable:** In this project, the dependent variable is the amount of rainfall.
- **Independent Variables:** Various factors that potentially influence rainfall are considered as predictors. These can include temperature, humidity, wind speed, atmospheric pressure, and other relevant meteorological data.

**The linear regression model is specified as:**

Rainfall=β 
0
​
 +β 
1
​
 Temperature+β 
2
​
 Humidity+β 
3
​
 Wind Speed+β 
4
​
 Pressure+ϵ
where 
𝛽
0
β 
0
​
  is the intercept, 
𝛽
1
,
𝛽
2
,
𝛽
3
,
𝛽
4
β 
1
​
 ,β 
2
​
 ,β 
3
​
 ,β 
4
​
  are the coefficients, and 
𝜖
ϵ is the error term.

#### Parameter Estimation
Once the model is specified, the next step is to estimate the parameters (coefficients) of the model. This is achieved using:

- **Ordinary Least Squares (OLS):** OLS is a method to estimate the parameters by minimizing the sum of the squared differences between the observed and predicted values. The objective is to find the best-fitting line that minimizes the residual sum of squares (RSS).

In the project, the OLS method is implemented using Python's scikit-learn library. The LinearRegression class is used to fit the model to the data:

```
from sklearn.linear_model import LinearRegression

Define the model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Extract the coefficients
coefficients = model.coef_
intercept = model.intercept_

```

Here, X represents the independent variables, and y is the dependent variable (rainfall).


#### Model Evaluation
Evaluating the performance of the linear regression model is crucial to ensure its accuracy and reliability. The project employs several metrics for this purpose:

- **R-squared (R²):** R² measures the proportion of the variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, with higher values indicating better model fit.
- **Mean Squared Error (MSE):** MSE is the average of the squared differences between the observed and predicted values. It provides a measure of the model's prediction accuracy.

The evaluation is performed using Python's scikit-learn library:

```

from sklearn.metrics import mean_squared_error, r2_score

# Make predictions
predictions = model.predict(X_test)

# Calculate R-squared
r_squared = r2_score(y_test, predictions)

# Calculate MSE
mse = mean_squared_error(y_test, predictions)

```

#### Implementation Steps in the Project
- **Data Collection and Preprocessing:** Collect and preprocess the historical weather data, ensuring that all relevant predictors are included and cleaned.
- **Feature Selection:** Identify and select the most significant predictors that influence rainfall. This can be done using correlation analysis or other feature selection techniques.
- **Model Specification:** Define the linear regression model with rainfall as the dependent variable and the selected predictors as independent variables.
- **Parameter Estimation:** Use the OLS method to estimate the model coefficients.
- **Model Training:** Train the linear regression model on the historical data.
- **Model Evaluation:** Evaluate the model using R-squared and MSE to ensure it accurately predicts rainfall.
- **Prediction:** Use the trained model to predict future rainfall based on the predictors.

By following these steps, the project effectively employs Linear Regression to uncover relationships between rainfall and various meteorological factors, providing valuable insights and accurate predictions.

### Support Vector Machine Regression (SVR)
Support Vector Machine Regression (SVR) is a powerful technique used in this project to handle nonlinear relationships in the rainfall data. SVR is particularly useful when the relationship between predictors and the dependent variable is complex and not well captured by linear models. Here’s a detailed breakdown of how SVR is utilized in this project:

#### Kernel Trick
The kernel trick is a fundamental aspect of SVR, allowing it to model complex, nonlinear relationships by transforming the input data into a higher-dimensional space where a linear separation is possible. This project leverages different kernel functions to capture the nonlinear patterns in the rainfall data:

- **Linear Kernel:** Suitable for linear relationships.
- **Polynomial Kernel:** Captures polynomial relationships by adding polynomial features.
- **Radial Basis Function (RBF) Kernel:** Most commonly used for capturing nonlinear patterns by mapping the data into an infinite-dimensional space.

In this project, the RBF kernel is primarily used due to its flexibility and ability to model complex nonlinear relationships effectively. The transformation is implemented using Python's scikit-learn library:

```

from sklearn.svm import SVR

# Define the model with RBF kernel
model = SVR(kernel='rbf')

# Fit the model to the data
model.fit(X_train, y_train)

```

Here, X_train represents the independent variables, and y_train is the dependent variable (rainfall).

#### Model Training
Training the SVR model involves finding the optimal hyperplane that maximizes the margin between support vectors (data points closest to the hyperplane). The SVR algorithm seeks to balance the model complexity and prediction error by minimizing the following objective function:

1
/
2
​
 ∣∣w∣∣ 
2
 +C∑ξ 
i
​


where 
𝑤
w represents the model parameters, 
𝜉
𝑖
ξ 
i
​
  are the slack variables for the margin of tolerance, and 
𝐶
C is the regularization parameter controlling the trade-off between margin maximization and error minimization.

In the project, the SVR model is trained using the training dataset, ensuring it generalizes well to unseen data:

```

# Train the SVR model
model.fit(X_train, y_train)

```

Here, the fit method optimizes the hyperplane to achieve the best prediction accuracy.

#### Hyperparameter Tuning
Hyperparameter tuning is crucial for improving the SVR model’s performance. Key parameters adjusted in this project include:

- **Regularization Parameter (C):** Controls the trade-off between achieving a low error on the training data and minimizing the model complexity. A higher value of C aims for a lower bias and higher variance, while a lower value aims for a higher bias and lower variance.
- **RKernel Parameters (e.g., gamma for RBF kernel):** Defines how far the influence of a single training example reaches. A lower gamma value means a larger reach, and a higher gamma value means a smaller reach.

The project employs Grid Search Cross-Validation to find the optimal hyperparameter values. This method systematically works through multiple combinations of parameter values, cross-validating as it goes to determine which combination gives the best performance:

```

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(SVR(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best parameters
best_params = grid_search.best_params_

# Best SVR model
best_model = grid_search.best_estimator_

```

Here, param_grid contains the hyperparameters to be tuned, and GridSearchCV performs the exhaustive search to identify the best combination.

#### Implementation Steps in the Project

- **Data Preprocessing:** Collect and preprocess the historical weather data, ensuring all relevant predictors are included and cleaned.
- **Kernel Selection:** Choose the appropriate kernel function (RBF in this case) to handle the nonlinear relationships.
- **Model Training:** Train the SVR model using the preprocessed training dataset.
- **Hyperparameter Tuning:** Apply Grid Search Cross-Validation to optimize the regularization parameter (C) and kernel parameter (gamma).
- **Model Evaluation:** Evaluate the model performance using metrics like Mean Squared Error (MSE) and R-squared (R²) to ensure it accurately predicts rainfall.
- **Prediction:** Use the tuned and trained SVR model to predict future rainfall based on the predictors.

By following these steps, the project effectively employs Support Vector Machine Regression to capture and predict complex nonlinear relationships in the rainfall data, providing accurate and reliable forecasts.


