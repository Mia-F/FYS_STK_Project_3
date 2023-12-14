import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import talib as ta
from NN_BTC_PROJ import Data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

#Import Bitcoin data and add technical indicators from NN_BTC_PROJ.py
data_frame = Data("/Users/miafrivik/Documents/GitHub/FYS_STK_Project_3/Data/BTC-USD_2014.csv")
data_frame.load_data()
data_frame.add_technical_indicators()
ta_data = data_frame.extract_data_for_NN()

#Creat define matrix and validation array
X = ta_data.drop(["Close", "Target"], axis=1)
y = ta_data.loc[:, "Close"]

#Define min and max depth for tree
n_max = 20
n_min = 1

depth = np.linspace(1, 20, 20).astype(int)
mse = np.zeros(len(depth))
R2_score = np.zeros(len(depth))
mse_bootstrap = np.zeros(len(depth))
R2_score_bootstap = np.zeros(len(depth))

#split in to train test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8)

#Scaleing the data
#scaler = MinMaxScaler(feature_range=(0,1))
scaler = StandardScaler()
scaler.fit(X_train)
    
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_scaled = (y_train - np.mean(y_train))/np.std(y_train)
y_test_scaled = (y_test - np.mean(y_train))/np.std(y_train)

#Creating an array with the criterions for DecisionTreeRegressor
methods = np.array(["squared_error", "friedman_mse","absolute_error"])

for i in tqdm(range(len(depth))):
    clf = RandomForestRegressor(criterion='absolute_error',  bootstrap=True, random_state=42, max_depth=depth[i])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    mse[i] = mean_squared_error(y_test, y_pred)
    R2_score[i] = r2_score(y_test, y_pred)
    
    clf = RandomForestRegressor(criterion='absolute_error',random_state=42, max_depth=depth[i], bootstrap=False)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    mse_bootstrap[i] = mean_squared_error(y_test, y_pred)
    R2_score_bootstap[i] = r2_score(y_test, y_pred)

plt.plot(depth, mse, label = "With bootstrap")
plt.plot(depth, mse_bootstrap, label = "without bootstrap")
plt.legend()
plt.show()

plt.plot(depth, R2_score, label = "With bootstrap")
plt.plot(depth, R2_score_bootstap, label = "without bootstrap")
plt.legend()
plt.show()



