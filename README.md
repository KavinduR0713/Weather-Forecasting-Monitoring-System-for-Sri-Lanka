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

#### Results of Time Series Analysis - SARIMA

##### Colombo District Actual Rainfall and Predicted Rainfall using Time Series Analysis - SARIMA

![Colombo_ TimeSeries_Forecasting_Values](https://github.com/KavinduR0713/Weather-Forecasting-Monitoring-System-for-Sri-Lanka/assets/105490780/413d0f59-fea2-43dc-8db3-b69ccf9c1c9b)

##### Time Series Analysis SARIMA Forecasting plot for Colombo district

![Colombo_ TimeSeries_Forecasting_Plot](https://github.com/KavinduR0713/Weather-Forecasting-Monitoring-System-for-Sri-Lanka/assets/105490780/0ce7ed96-cbf3-4f98-b6f6-31b333c00255)

##### Comparison of Actual rainfall and Predicted Rainfall in Colombo District

**1. Year 2021 rainfall Comparison between Actual rainfall and Predicted rainfall.**

![2021_Time Series](https://github.com/KavinduR0713/Weather-Forecasting-Monitoring-System-for-Sri-Lanka/assets/105490780/903c8a17-9acf-43d8-b1b2-e190ecfd3c63)

**1. Year 2022 rainfall Comparison between Actual rainfall and Predicted rainfall.**

![2022_Time Series](https://github.com/KavinduR0713/Weather-Forecasting-Monitoring-System-for-Sri-Lanka/assets/105490780/9e647a86-5435-46e4-8286-dc233cf04060)

**1. Year 2023 rainfall Comparison between Actual rainfall and Predicted rainfall.**

![2023_Time Series](https://github.com/KavinduR0713/Weather-Forecasting-Monitoring-System-for-Sri-Lanka/assets/105490780/3b652159-cdfd-4690-9730-f5165990d379)


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

#### Results of Linear Regression Analysis

##### Anuradhapura District Actual Rainfall and Predicted Rainfall using Linear Regression Analysis
![Anuradhapura_Regression_Forecasting_Value](https://github.com/KavinduR0713/Weather-Forecasting-Monitoring-System-for-Sri-Lanka/assets/105490780/ceca5537-9935-4e5f-8d39-e21427c5bb1a)

##### Scatter plot for Anuradhapura district
- scatter plot shows the relationship between Anuradhapura district’s Actual Rainfall and predicted Rainfall using Linear Regression Analysis.
![Anuradhapura_Regression_Forecasting_Plot](https://github.com/KavinduR0713/Weather-Forecasting-Monitoring-System-for-Sri-Lanka/assets/105490780/2a6f45e9-2ad2-4b51-a90e-48054d4470e5)

##### Comparison of Actual rainfall and Predicted Rainfall in Anuradhapura District

**1. Year 2021 rainfall Comparison between Actual rainfall and Predicted rainfall.**

![2021_Regression](https://github.com/KavinduR0713/Weather-Forecasting-Monitoring-System-for-Sri-Lanka/assets/105490780/de7a35cf-9364-4127-9a2d-c606cffc0688) 

**2. Year 2022 rainfall Comparison between Actual rainfall and Predicted rainfall.**

![2022_Regression](https://github.com/KavinduR0713/Weather-Forecasting-Monitoring-System-for-Sri-Lanka/assets/105490780/7e2685c1-f47c-4303-83d1-7b03d1541899)

**3. Year 2023 rainfall Comparison between Actual rainfall and Predicted rainfall.**

![2023_Regression](https://github.com/KavinduR0713/Weather-Forecasting-Monitoring-System-for-Sri-Lanka/assets/105490780/de3c89ef-7fbb-42d7-ac4b-030466ef16a2) 


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

#### Results of Support Vector Machine Regression analysis

##### Jaffna District Actual Rainfall and Predicted Rainfall using Support Vector Machine Regression analysis.

![Jaffna_RandomForestRegressor_Value](https://github.com/KavinduR0713/Weather-Forecasting-Monitoring-System-for-Sri-Lanka/assets/105490780/29269071-4df7-4ae7-ba99-d3b1a1bf4108)

##### Comparison of Actual rainfall and Predicted Rainfall in Jaffna District 

**1.Year 2021 rainfall Comparison between Actual rainfall and Predicted rainfall** 

![2021 SVMR](https://github.com/KavinduR0713/Weather-Forecasting-Monitoring-System-for-Sri-Lanka/assets/105490780/e7900897-05f5-480b-8e98-5975436bc794)

**1.Year 2022 rainfall Comparison between Actual rainfall and Predicted rainfall**

![2022 SVMR](https://github.com/KavinduR0713/Weather-Forecasting-Monitoring-System-for-Sri-Lanka/assets/105490780/a596b05d-1e07-490d-bfc4-80bec8b0b95c)

**1.Year 2023 rainfall Comparison between Actual rainfall and Predicted rainfall**

![2023 SVMR](https://github.com/KavinduR0713/Weather-Forecasting-Monitoring-System-for-Sri-Lanka/assets/105490780/9a1c1c4a-b856-4377-980b-f6005824feeb)

### Mean Squared Error (MSE) Analysis
MSE is the average of the squared differences between the observed actual outcomes and the outcomes predicted by the model. It is a critical measure of a model's accuracy, with lower values indicating better performance.

#### MSE Comparison
- **SARIMA:** The MSE values for SARIMA were the lowest across all districts, indicating its superior ability to predict rainfall accurately. The model's consideration of both trend and seasonality contributed to its minimal prediction errors

- **Linear Regression:** The MSE values were higher compared to SARIMA, reflecting the model's limitations in capturing complex seasonal patterns. This resulted in larger discrepancies between the observed and predicted rainfall values.

- **SVR:** While SVR showed better performance than Linear Regression in some regions, its MSE values were still higher than those of SARIMA. SVR's ability to handle nonlinear relationships improved its accuracy, but it was not as consistent as SARIMA across all districts.

#### Regional Performance
A detailed analysis of MSE across different districts further highlights the effectiveness of each model:
   - **Coastal Regions:** SARIMA achieved the lowest MSE, accurately predicting seasonal rainfall patterns influenced by monsoons.
   - **Inland Regions:** Both SARIMA and SVR performed well, but SARIMA maintained a lower MSE, especially in regions with distinct seasonal trends.
   - **Overall:** The consistency of low MSE values for SARIMA across various districts established it as the most reliable model for rainfall prediction in Sri Lanka.

The comprehensive comparison and analysis using MSE clearly indicate that the SARIMA model is the most effective tool for predicting rainfall in Sri Lanka. Its ability to accurately model both trend and seasonality makes it the primary choice for future predictions. The detailed visualizations and statistical metrics provide strong evidence of SARIMA's superior performance, ensuring reliable and accurate rainfall forecasts.

The project's results underscore the importance of choosing the right model for time series forecasting, and SARIMA's robustness in handling the unique characteristics of rainfall data in Sri Lanka makes it an invaluable tool for stakeholders in agriculture, resource management, and disaster preparedness.

![MSE](https://github.com/KavinduR0713/Weather-Forecasting-Monitoring-System-for-Sri-Lanka/assets/105490780/7d192e97-9448-4dca-8bc7-3332ca20fbcd)
