"""
This document contains the calculations for Bitcoin predictions with Random forest with and without bagging
The code in this document are based on the following codes:
https://www.kaggle.com/code/rishidamarla/stock-market-prediction-using-decision-tree
https://www.kaggle.com/code/prashant111/decision-tree-classifier-tutorial
"""

import numpy as np
import seaborn as sns
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
import random

#Setting a seed
random.seed(2023)

#Setting some plotting parameters
sns.set_theme()
params = {
    "font.family": "Serif",
    "font.serif": "Roman", 
    "text.usetex": True,
    "axes.titlesize": "large",
    "axes.labelsize": "large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
    "legend.fontsize": "large"
}

plt.rcParams.update(params)
pd.options.mode.chained_assignment = None  # default='warn'

#Setting path for figures
cwd = os.getcwd()
path = Path(cwd) / "FigurePlots" / "Random_forest"

if not path.exists():
    path.mkdir()


def predict_future(data_frame=None, depth=3, predicted_days = 10, method="squared_error", bootstrap = False):
    """
    Description:
    ------------
    A class that uses random forest to predict the closing price of Bitcoins.
    This function is based on the following code: https://www.kaggle.com/code/rishidamarla/stock-market-prediction-using-decision-tree
    Parameters:
    ------------
        I   data_frame (pd.data_frame): A pandas data frame containing the parameters used to make a model
        II  depth (int): The max depth of the decision tree
        III predicted_days (int): The amount of days the price is predicted 
        IV  method (str): a string of either squared_error, friedman_mse, absolute error or poisson that is used as criterion for DecisionTreeRegressor  
        V   splitter (str): string of either "best" or "random" which is the strategy used to choose the split at each node 
    Returns:
        I   valid (pd.data_frame): A pandas data frame containing the data for the predicted_days including the predicted prices 
        II  days (np.ndarray): An array containing at least the 100 last day in the dataset.    
    ------------       
    """
    df = pd.DataFrame()
    df = data_frame 
    #drop the label colum in datset
    df = df.drop(["Label"], axis=1)
    #Include a prediction colum that is the price of Bitcoin when market close for all days expet the prediction days
    df["Prediction"] = df["Close"].shift(-predicted_days)
    #removes the prediction for the prediction days from data set
    X = df.drop(["Prediction"], axis=1)[:-predicted_days]
    y = df["Prediction"][:-predicted_days]

    #uses the datset to creat a model 
    clf = RandomForestRegressor(criterion=method,  bootstrap=bootstrap, random_state=42, max_depth=depth)
    #Fit model with data tath does not include the predicted days
    clf.fit(X, y)

    #creat data_frem with predicted days
    x_future = df.drop(["Prediction"], axis = 1)[-predicted_days:]
    x_future = x_future.tail(predicted_days)
        
    # Predict prices of future days
    tree_prediction = clf.predict(x_future)
    predictions = tree_prediction 
        
    #Adds the result for predicted day in a separate data_frame
    valid = x_future
    valid["Prediction"] = predictions

    #Plott result against actuall predicted prices
    lenght = len(X.loc[:,"Close"])
    days = np.linspace(lenght + 1 + 108, lenght + predicted_days + 1 + 108, predicted_days)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlabel("Days")
    ax.set_ylabel("US Dollers")
    ax.plot(data_frame.loc[:,"Close"], "--", label="Actual closing price")
    ax.scatter(days, valid.loc[:,"Prediction"], label="Predicted closing price", color="tab:orange")
    ax.legend()
    plt.savefig(path / f"predicted_prices_for_{predicted_days}_days.png")
    plt.close()

    #if predicted days is less than 100, the prediction still gets plotted with 100 days.
    if predicted_days <= 100:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_xlabel("Days")
        ax.set_ylabel("US Dollers")
        ax.plot(data_frame.loc[3187:,"Close"], "--", label="Actual closing price")
        ax.scatter(days, valid.loc[:,"Prediction"], label="Predicted closing price", color="tab:orange")
        ax.legend()
        plt.savefig(path / f"predicted_prices_for_{predicted_days}_days_zoomed.png")
        plt.close()
    
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_xlabel("Days")
        ax.set_ylabel("US Dollers")
        ax.plot(data_frame.loc[days[0]:,"Close"], "--", label="Actual closing price")
        ax.scatter(days, valid.loc[:,"Prediction"], label="Predicted closing price", color="tab:orange")
        ax.legend()
        plt.savefig(path / f"predicted_prices_for_{predicted_days}_days_zoomed.png")
        plt.close()
    return valid, days

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

depth = np.linspace(1, 10, 10).astype(int)

#split in to train test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8)

#Scaling the data
scaler = StandardScaler()
scaler.fit(X_train)
    
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_scaled = (y_train - np.mean(y_train))/np.std(y_train)
y_test_scaled = (y_test - np.mean(y_train))/np.std(y_train)

#Creating an array with the criterions for DecisionTreeRegressor
methods = np.array(["squared_error", "friedman_mse","absolute_error"])
bootstrap = np.array([False, True])

#Define arrays for mse and R2 score
mse = np.zeros((len(methods), len(bootstrap), len(depth)))
R2_score = np.zeros((len(methods), len(bootstrap), len(depth)))

j = 0
k = 0
for method in methods:
    print(f"Method = {method}")
    k = 0
    for boot in bootstrap:
        print(f"Bootstrap = {boot}")
        for i in tqdm(range(len(depth))):
                #Uses the datset to creat a model 
                clf = RandomForestRegressor(criterion=method,  bootstrap=boot, random_state=42, max_depth=depth[i])
                #Fit model with data 
                clf.fit(X_train_scaled, y_train_scaled)
                # Predict prices
                y_pred = clf.predict(X_test_scaled)

                #Store mse and R2 score values 
                mse[j][k][i] = mean_squared_error(y_test_scaled, y_pred)
                R2_score[j][k][i] = r2_score(y_test_scaled, y_pred)
        k += 1
    j += 1

#Plotting the mse  as a function of depth
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(depth, mse[0][0][:], label="Square error witout bootstrap")
ax.plot(depth, mse[0][1][:], label="Square error with boostrap", linestyle='dashed')
ax.plot(depth, mse[1][0][:], label="Friedman MSE witout bootstrap")
ax.plot(depth, mse[1][1][:], label="Friedman MSE with boostrap", linestyle='dashed')
ax.plot(depth, mse[2][0][:], label="Absolute error witout bootstrap")
ax.plot(depth, mse[2][1][:], label="Absolute error With boostrap", linestyle='dashed')
ax.legend()
ax.set_xlabel("Depth")
ax.set_ylabel("MSE")
plt.savefig(path / "mse_different_methods_random_forest.png")
plt.close()

#Plotting th R2 score as a function of depth
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(depth, R2_score[0][0][:], label="Square error witout bootstrap")
ax.plot(depth, R2_score[0][1][:], label="Square error with boostrap")
ax.plot(depth, R2_score[1][0][:], label="Friedman MSE witout bootstrap")
ax.plot(depth, R2_score[1][1][:], label="Friedman MSE with boostrap")
ax.plot(depth, R2_score[2][0][:], label="Absolute error witout bootstrap")
ax.plot(depth, R2_score[2][1][:], label="Absolute error With boostrap")
ax.set_xlabel("Depth")
ax.set_ylabel("R2 score")
ax.legend()
plt.savefig(path / "r2_score_different_methods_random_forest.png")
plt.close()


print("\n")
print("----------------------------------- Results -----------------------------------")
print("Without bootstrap")
print(f"The lowest MSE obtained without bootstrap with square error was: {np.min(mse[0][0][:]):.5f}")
print(f"The lowest MSE obtained without bootstrap with Friedman MSE was: {np.min(mse[1][0][:]):.5f}")
print(f"The lowest MSE obtained without bootstrap with absolute error was: {np.min(mse[2][0][:]):.5f}")
print("With bootstrap")
print(f"The lowest MSE obtained with bootstrap with square error was: {np.min(mse[0][1][:]):.5f}")
print(f"The lowest MSE obtained with bootstrap with Friedman MSE was: {np.min(mse[1][1][:]):.5f}")
print(f"The lowest MSE obtained with bootstrap with absolute error was: {np.min(mse[2][1][:]):.5f}")
print("-------------------------------------------------------------------------------")

#Creating bar plots of the lowest mse achived with each criterions with bootstrap
fig, ax = plt.subplots(figsize=(9, 5))
values = np.array([np.min(mse[0][0][:]), np.min(mse[1][0][:]), np.min(mse[2][0][:])])
ax.bar(methods, values,  align="center")
ax.set_ylabel("Lowest MSE")
plt.savefig(path / "bar_plot_no_bootstrap.png")
plt.close()

#Creating bar plots of the lowest mse achived with each criterions without bootstrap
fig, ax = plt.subplots(figsize=(9, 5))
values = np.array([np.min(mse[0][1][:]), np.min(mse[1][1][:]), np.min(mse[2][1][:])])
ax.bar(methods, values,  align="center")
ax.set_ylabel("Lowest MSE")
plt.savefig(path / "bar_plot_bootstrap.png")
plt.close()

#Get the index for where the lowest mse value obtained
i,j,k = np.where(mse == mse.min())

#Find which parameters gave the lowest mse
best_method = methods[i[0]]
best_boot = bootstrap[j][0]
best_depth = depth[k][0]


#Creating an array of days we want to predict Bitcoin prices
predicted_days = np.array([1, 10, 100, 365])

for i in range(len(predicted_days)):
    #Predict Bitcoin closing prices
    valid, days = predict_future(data_frame=ta_data, depth=best_depth, predicted_days = predicted_days[i], method=best_method, bootstrap = best_boot)
    if predicted_days[i] == 1:
        #Plott the Difference between actual closing price and predicted one
        fig, ax = plt.subplots(figsize=(12,6))
        ax.scatter(days, ta_data.loc[days[0]:,"Close"] - valid.loc[:,"Prediction"])
        ax.set_xlabel("Days since 15 november 2014")
        ax.set_ylabel("Difference between actual and predicted price")
        plt.savefig(path / f"Difference_in_predicted_prices_for_{predicted_days[i]}_days.png")
        plt.close()
    
    else:
        #Plott the Difference between actual closing prices and predicted ones
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(days, ta_data.loc[days[0]:,"Close"] - valid.loc[:,"Prediction"])
        ax.set_xlabel("Days since 15 november 2014")
        ax.set_ylabel("Difference between actual and predicted price")
        plt.savefig(path / f"Difference_in_predicted_prices_for_{predicted_days[i]}_days.png")
        plt.close()
