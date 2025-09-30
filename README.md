# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 30-9-2025

#### NAME:HCYINTH D
#### REGISTER NUMBER:21222324005

### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import seaborn as sns

# Load dataset from URL
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
data = pd.read_csv(url)

# Convert date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Set as index
data.set_index('Date', inplace=True)

# Keep only the numeric column you want to analyze
ts = data['Temp']

# ADF Test
result = adfuller(ts.dropna())
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])

# Split into train/test
x = int(0.8 * len(data))
train_data = data.iloc[:x]
test_data = data.iloc[x:]

# Plot ACF and PACF
plot_acf(ts.dropna(), lags=30)
plot_pacf(ts.dropna(), lags=30)
plt.show()

# Split series into train and test
train_size = int(len(ts) * 0.8)
train, test = ts[0:train_size], ts[train_size:]

# Fit AR model (ensure lag < number of training points)
lags = min(5, len(train)-1)
model = AutoReg(train, lags=lags).fit()
print(model.summary())

# Forecast
preds = model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)

# Calculate MSE
error = mean_squared_error(test, preds)
print("MSE:", error)

# Plot actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, preds, label='Predicted', color='red')
plt.legend()
plt.show()


```
### OUTPUT:

### GIVEN DATA

<img width="272" height="250" alt="image" src="https://github.com/user-attachments/assets/ec1445b6-1809-47b4-90dd-ebb0e138d056" />

### PACF - ACF

<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/225ca225-88e8-4bbb-a6e2-b8c2cb9288cb" />

<img width="568" height="435" alt="image" src="https://github.com/user-attachments/assets/4eb7cf69-e5e8-43d9-a176-c9eb0d932c44" />

###  PREDICTION

<img width="422" height="177" alt="image" src="https://github.com/user-attachments/assets/0b5a7f6d-3904-4448-80d3-148ae42d1798" />


### FINIAL PREDICTION

<img width="822" height="428" alt="image" src="https://github.com/user-attachments/assets/372f4d35-b704-4abd-ad4b-7a9744778a00" />


### RESULT:
Thus we have successfully implemented the auto regression function using python.
