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
from NN_BTC_PROJ import Data

fontsize = 29
sns.set_theme()
params = {
    "font.family": "Serif",
    "font.serif": "Roman", 
    "text.usetex": True,
    "axes.titlesize": fontsize,
    "axes.labelsize": fontsize,
    "xtick.labelsize": "x-large",
    "ytick.labelsize": "x-large",
    "legend.fontsize": fontsize
}

plt.rcParams.update(params)
pd.options.mode.chained_assignment = None  # default='warn'

def testing_decision_tress(method, n_min, n_max, path):
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
            y_pred, y_pred_train = Model.predict()

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
    
    #Plot result and save figurs for MSE and R2 score
    fig, ax = plt.subplots()
    m = ax.imshow(mse,extent=[0,10,n_max,n_min])
    ax.set_aspect('equal')
    fig.colorbar(m, label="MSE")
    ax.set_xlabel("Random state")
    ax.set_ylabel("Depth")
    plt.savefig(path / "mse.png")
    plt.close()

    fig, ax = plt.subplots()
    m = ax.imshow(mse_train_2D, extent=[0,10,n_max,n_min])
    fig.colorbar(m, label="MSE")
    ax.set_xlabel("Random state")
    ax.set_ylabel("Depth")
    plt.savefig(path / "mse_train.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(depth, mse[:,lowest_mse_index[1]])
    ax.set_xlabel("Depth")
    ax.set_ylabel("MSE")
    plt.savefig(path / "mse_depth.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(depth, r2[:,lowest_mse_index[1]])
    ax.set_xlabel("Depth")
    ax.set_ylabel("R2 score")
    plt.savefig(path / "r2_depth.png")
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(depth, r2_train_2D[:,lowest_mse_index[1]])
    ax.set_xlabel("Depth")
    ax.set_ylabel("R2 score")
    plt.savefig(path / "r2_depth_train.png")
    plt.close()

    return lowest_mse_index , mse

#Import Bitcoin data and add technical indicators from NN_BTC_PROJ.py
data_frame = Data("/Users/miafrivik/Documents/GitHub/FYS_STK_Project_3/Data/BTC-USD_2014.csv")
data_frame.load_data()
data_frame.add_technical_indicators()
ta_data = data_frame.extract_data_for_NN()

#Creat define matrix and validation array
X = ta_data.drop(["Target"], axis=1)
y = ta_data.loc[:, "Target"]

#Define min and max depth for tree
n_max = 10
n_min = 4

#split in to train test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8)
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(X_train)
    
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train_scaled = (y_train - np.mean(y_train))/np.std(y_train)
y_test_scaled = (y_test - np.mean(y_train))/np.std(y_train)

methods = np.array(["squared_error", "friedman_mse","absolute_error"])

lowest_mse= []
mse = []

i=0
for method in methods:
    print(i)
    print(f"method = {method}")
    cwd = os.getcwd()
    path = Path(cwd) / "FigurePlots" / "Decision_tree"/ method
    if not path.exists():
        path.mkdir()
    
    lowest_mse_method, mse_method = testing_decision_tress(method,n_min, n_max, path)
    Model = Decision_tree(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, printing=True, depth=lowest_mse_method[0], randomnes=lowest_mse_method[1], alpha=True, method = method )
    Model.predict()
    lowest_mse.append(lowest_mse_method)
    index = lowest_mse_method[1]
    index = int(index)
    mse.append(mse_method[:,index-1])
    i += 1

mse = np.array(mse)

path = Path(cwd) / "FigurePlots" / "Decision_tree"
if not path.exists():
    path.mkdir()

depth = np.linspace(n_min, n_max, n_max-n_min).astype(int)
fig, ax = plt.subplots()
ax.plot(depth, mse[0,:], label="Square error")
ax.plot(depth, mse[1,:], label="Friedman mse")
ax.plot(depth, mse[2,:], label="Absolute error")
ax.legend()
plt.savefig(path / "mse_different_methods.png")
plt.close()


prediciton_model = predict_future_tree(X, y , printing=False, data_frame=ta_data, depth=4, predicted_days = 100, randomnes = 20, method = "friedman_mse")
prediciton_model.predict()
