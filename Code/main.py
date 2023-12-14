import numpy as np 
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
from tqdm import tqdm
import graphviz 
from Decision_Tree import Decision_tree, predict_future_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
from pathlib import Path
import talib as ta
from os import system
from NN_BTC_PROJ import Data
from sklearn import tree
import random 

random.seed(2023)

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

def testing_decision_tress(method, n_min, n_max, path, print):
    """
    Description:
    ------------
    A function that runs throu 10 random state values and n_max - n_min different depth and returns the MSE as a function of depth and
    random stat. Also returns the depth and random state that gave the lowest MSE. This function also save figurs for MSE and R2 score

    Parameters:
    ------------
        I   method (str): criterion used in DecisionTreeRegressor
        II  n_min (int): min depth of tree 
        III n_max (int): max depth of tree
        VI  Path: path for saving figure
        V   print (Boolean): If True figure of the best decision tree will be plotted, takes about 20 minutes, default False

    Returns:
    ------------
        I   lowest_mse_index (tuple): returns a tuple in the following format [best_depth, best_random_state]
        I   mse (np.ndarray): returns an array with the mse values as a function of depth and random state with dimension (len(depth), len(random_state))
    """

    #Start by creating the depth and random state arrays
    depth = np.linspace(n_min, n_max, n_max - n_min).astype(int)
    randomnes = np.linspace(0, 10, 11).astype(int)
    
    # Creating arrays for mse and r2 score
    mse = np.zeros((len(depth), len(randomnes)))
    mse_train_2D = np.zeros((len(depth), len(randomnes)))
    r2 = np.zeros((len(depth), len(randomnes)))
    r2_train_2D = np.zeros((len(depth), len(randomnes)))

    #create a tuple that will contain best depth and random state and also set a lowest mse
    lowest_mse = 10e10
    lowest_mse_index = [0,0]

    #run throug different depth and random states
    for i in tqdm(range(len(depth))):
        for j in range(len(randomnes)):

            #Call the Decision tree class from Decision_Tree.py 
            Model = Decision_tree(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, printing=False, depth=depth[i], randomnes=randomnes[j], method=method)
            y_pred, y_pred_train, parm = Model.predict()

            #Uptain the MSE and R2 score for both testing and training data
            mse_test, mse_train = Model.mse(y_pred, y_pred_train)
            r2_test, r2_train = Model.R2_score(y_pred, y_pred_train)

            mse[i][j] = mse_test
            mse_train_2D[i][j] = mse_train
            r2[i][j] = r2_test
            r2_train_2D[i][j] = r2_train
            
            # Uptain depth and random state with the lowest MSE
            if mse[i][j] < lowest_mse:
                lowest_mse = mse[i][j]
                lowest_mse_index = [depth[i],randomnes[j]]
                if print == True:
                    #creat a visulasation of the best tree
                    plt.figure(figsize=(12,8))
                    tree.plot_tree(parm) 
                    plt.savefig(path / "best_tree.png")
                    plt.close()

    

    
    #Plot result and save figurs for MSE and R2 score
    fig, ax = plt.subplots(figsize=(12,6))
    m = ax.imshow(mse,extent=[0,10,n_max,n_min])
    ax.set_aspect('equal')
    fig.colorbar(m, label="MSE")
    ax.set_xlabel("Random state")
    ax.set_ylabel("Depth")
    plt.savefig(path / "mse.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(12,6))
    m = ax.imshow(mse_train_2D, extent=[0,10,n_max,n_min])
    fig.colorbar(m, label="MSE")
    ax.set_xlabel("Random state")
    ax.set_ylabel("Depth")
    plt.savefig(path / "mse_train.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(depth, mse[:,lowest_mse_index[1]], label="Test")
    ax.plot(depth, mse_train_2D[:,lowest_mse_index[1]], label="Train")
    ax.set_xlabel("Depth")
    ax.set_ylabel("MSE")
    ax.legend()
    plt.savefig(path / "mse_depth.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(depth, r2[:,lowest_mse_index[1]])
    ax.set_xlabel("Depth")
    ax.set_ylabel("R2 score")
    plt.savefig(path / "r2_depth.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(depth, r2_train_2D[:,lowest_mse_index[1]])
    ax.set_xlabel("Depth")
    ax.set_ylabel("R2 score")
    plt.savefig(path / "r2_depth_train.png")
    plt.close()

    return lowest_mse_index , mse

#Import Bitcoin data and add technical indicators from NN_BTC_PROJ.py
data_frame = Data("./Data/BTC-USD_2014.csv")
data_frame.load_data()
data_frame.add_technical_indicators()
ta_data = data_frame.extract_data_for_NN()

#Creat define matrix and validation array
X = ta_data.drop(["Close", "Target"], axis=1)
y = ta_data.loc[:, "Close"]

#Define min and max depth for tree
n_max = 20
n_min = 1

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

lowest_mse= []
mse = []

i=0

#Runs through the 3 diffrent criterions and 10 diffrent random states
for method in methods:
    print(f"method = {method}")

    #Creat path for figurs
    cwd = os.getcwd()
    path = Path(cwd) / "FigurePlots" / "Decision_tree"/ method
    if not path.exists():
        path.mkdir()
    
    #Get the mse for the diffrent method and at witch depth and random state the lowest mse value is uptainded for each method
    lowest_mse_method, mse_method = testing_decision_tress(method,n_min, n_max, path, False)
    #Se how it is affected by pruning the tree
    Model = Decision_tree(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, printing=True, depth=lowest_mse_method[0], randomnes=lowest_mse_method[1], alpha=True, method = method )
    Model.predict()

    lowest_mse.append(lowest_mse_method)
    index = lowest_mse_method[1]
    index = int(index)
    mse.append(mse_method[:,index-1])
    i += 1

#change list to array
mse = np.array(mse)
lowest_mse = np.array(lowest_mse)

print("\n")
print("----------------------------------- Results -----------------------------------")
print(f"{methods[0]} had a lowest mse of {np.min(mse[0,:]):.5f}, when depth = {lowest_mse[0][0]} and random stat = {lowest_mse[0,1]}")
print(f"{methods[1]} had a lowest mse of {np.min(mse[1,:]):.5f}, when depth = {lowest_mse[1][0]} and random stat = {lowest_mse[1,1]}")
print(f"{methods[2]} had a lowest mse of {np.min(mse[2,:]):.5f}, when depth = {lowest_mse[2][0]} and random stat = {lowest_mse[2,1]}")
print("-------------------------------------------------------------------------------")

path = Path(cwd) / "FigurePlots" / "Decision_tree"
if not path.exists():
    path.mkdir()


depth = np.linspace(n_min, n_max, n_max-n_min).astype(int)
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(depth, mse[0,:], label="Square error")
ax.plot(depth, mse[1,:], label="Friedman mse")
ax.plot(depth, mse[2,:], label="Absolute error")
ax.set(ylim=(3, 15))
ax.legend()
ax.set_xlabel("Depth")
ax.set_ylabel("MSE")
plt.savefig(path / "mse_different_methods.png")
plt.close()

#Find which method that had the lowest mse and uptain an index to use for prediction
index = np.where(mse == np.min(mse))
method = methods[index[0]][0]
random_state = lowest_mse[index[0],1][0]

#Use the best model found to predict Bitcoin prices
prediciton_model = predict_future_tree(data_frame=ta_data, depth=lowest_mse[index[0],0], predicted_days = 100, randomnes = random_state , method = method)
prediciton_model.predict()

