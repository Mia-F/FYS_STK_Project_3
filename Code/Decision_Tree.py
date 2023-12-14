"""
This documnet contains the two classes Decision_tree and predict_future_tree
The code in this document are based on the following codes:
https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
https://www.kaggle.com/code/rishidamarla/stock-market-prediction-using-decision-tree
https://www.kaggle.com/code/prashant111/decision-tree-classifier-tutorial
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from os import system
import random
import os
from pathlib import Path
import pandas as pd 


class Decision_tree:
    def __init__(self,  X_train, y_train, X_test, y_test, printing=False, depth=3, randomnes = 0, alpha= False, method="squared_error") -> None:
        """
        Description:
        ------------
        A class that uses decision tree to create a model of closing prices of Bitcoins
        The code in this function is based on the "Decision-Tree Classifier Tutorial" from https://www.kaggle.com/code/prashant111/decision-tree-classifier-tutorial
        and the part with pruning (alpha) is taken from sckit learns example "Post pruning decision trees with cost complexity pruning" from https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html

        Parameters:
        ------------
            I    X_train (pd.data_frame): A pandas dataframe containing the different parameters the model is trained on.
            II   Y_train (pd.data_frame): A pandas dataframe containing the targets the model is trained on. (for these runs it is set to the closing time)
            III  X_test (pd.data_frame): Same as X_train only with the test data
            IV   Y_test (pd.data_frame): Same as y_train only with the test data
            V    printing (Boolean): If True the class will print results while running, default False
            VI   depth (int): The max depth of the decision tree
            VII  randomnes (int): The random state used in DecisionTreeRegressor
            VIII alpha (Boolean): If True the prediction function will plot dependencies on alpha (pruning) and not return y_pred, y_pred_test and clf, default Flase
            IX   method (str): a string of either squared_error, friedman_mse, absolute error or poisson that is used as criterion for DecisionTreeRegressor  
        functions:
            I    predict:  If alpha is False, the function returns y_pred and y_pred_test which is the predictions for this model.
                           If alpha is True, the dependencis for alpha is plotted.
            II   R2_score: A function that calculates the R2 score for the test and train data and returns these.
            III  mse:      A function that calculates the MSE values for test and train and returns these. 
        ------------       
        """
        self.depth = depth
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.printing = printing
        self.randomnes = randomnes
        self.alpha = alpha
        self.method = method
    
    def predict(self):
        """
        Description:
        ------------
            A function using skit-learns decisonTreeRegressor to fit the model to the training dataset 
        Returns:
        ------------
            I  y_pred (np.ndarray): An array containing the model of the Bitcoin closing prices of the test data  
            II y_pred_train (np.ndarray): An array containing the model of the Bitcoin closing prices of the train data     
            III clf: 
        ------------
        """
    
        clf = DecisionTreeRegressor(criterion=self.method, max_depth=self.depth, random_state=self.randomnes)
        if self.alpha == True:
            """
            The code for pruning is taken from https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html
            """

            #Define path for saving plots
            cwd = os.getcwd()
            
            path = Path(cwd) / "FigurePlots" / "Decision_tree"/ self.method 
            if not path.exists():
                path.mkdir()

            path_pruning = clf.cost_complexity_pruning_path(self.X_train, self.y_train)
            ccp_alphas, impurities = path_pruning.ccp_alphas, path_pruning.impurities
            clfs = []

            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
            ax.set_xlabel("effective alpha")
            ax.set_ylabel("total impurity of leaves")
            plt.savefig(path / "Impurity.png")
            plt.close()

            mse = np.zeros(len(ccp_alphas))
            mse_train = np.zeros(len(ccp_alphas))

            for i in range(len(ccp_alphas)):
                clf = DecisionTreeRegressor(criterion=self.method, random_state=0, ccp_alpha=ccp_alphas[i])
                clf.fit(self.X_train, self.y_train)
                clfs.append(clf)
                y_pred = clf.predict(self.X_test)
                y_pred_train = clf.predict(self.X_train)

                mse[i] = mean_squared_error(self.y_test, y_pred)
                mse_train[i] = mean_squared_error(self.y_train, y_pred_train)

            clfs = clfs[:-1]
            ccp_alphas = ccp_alphas[:-1]

            node_counts = [clf.tree_.node_count for clf in clfs]
            depth = [clf.tree_.max_depth for clf in clfs]
            fig, ax = plt.subplots(2, 1, figsize=(12,6))
            ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
            ax[0].set_xlabel("alpha")
            ax[0].set_ylabel("number of nodes")
            ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
            ax[1].set_xlabel("alpha")
            ax[1].set_ylabel("depth of tree")
            plt.savefig(path / "Alpha_depth.png")
            plt.close()

            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(ccp_alphas, mse[:-1], label="Test")
            ax.plot(ccp_alphas, mse_train[:-1], label="Train")
            ax.set_xlabel("alpha")
            ax.set_ylabel("MSE")
            ax.legend()
            plt.savefig(path / "Alpha_mse.png")
            plt.close()
                    
        else:   
            #Fit model with training data
            parm = clf.fit(self.X_train, self.y_train)

            #Predict a model with test data
            y_pred = clf.predict(self.X_test)
            #Predict a model with training data
            y_pred_train = clf.predict(self.X_train)
            return y_pred, y_pred_train, parm
        
    def R2_score(self, y_pred, y_pred_train):
        """
        Description:
        ------------
            A function that calculates the R2 score 
        Parameters:
        ------------ 
            I   y_pred (np.ndarray): An array containing the model of the Bitcoin closing prices of the test data  
            II  y_pred_train (np.ndarray): An array containing the model of the Bitcoin closing prices of the train data
        Returns:
        ------------
            I   r2_score_test (float): The R2 score of the model based on test data
            II  r2_score_train (float): The R2 score of the model based on training data     
        ------------
        """
        r2_score_test = r2_score(self.y_test, y_pred)
        r2_score_train = r2_score(self.y_train, y_pred_train)
        if self.printing==True:
            print(f"Model R2 score: {r2_score_test:.4f}")
            print(f"Training-set R2 score: {r2_score_train:.4f}")
        return r2_score_test, r2_score_train
    
    def mse(self, y_pred, y_pred_train):
        """
        Description:
        ------------
            A function that calculates the MSE 
        Parameters:
        ------------ 
            I   y_pred (np.ndarray): An array containing the model of the Bitcoin closing prices of the test data  
            II  y_pred_train (np.ndarray): An array containing the model of the Bitcoin closing prices of the train data
        Returns:
        ------------
            I   r2_score_test (float): The MSE of the model based on test data
            II  r2_score_train (float): The MSE of the model based on training data     
        ------------
        """
        mse_test = mean_squared_error(self.y_test, y_pred)
        mse_train = mean_squared_error(self.y_train, y_pred_train)
        if self.printing==True:
            print(f"Model MSE value: {mse_test:.4f}")
            print(f"Training-set MSE value: {mse_train:.4f}")
        return mse_test, mse_train
    


class predict_future_tree:
    def __init__(self , data_frame=None, depth=3, predicted_days = 10, randomnes = 0, method="squared_error") -> None:
        """
        Description:
        ------------
        A class that uses decision tree to predict the closing price of Bitcoins
        Parameters:
        ------------
            I   data_frame (pd.data_frame): A pandas data frame containing the parameters used to make a model
            II  depth (int): The max depth of the decision tree
            III predicted_days (int): The amount of days the price is predicted 
            IV  randomnes (int): The random state used in DecisionTreeRegressor
            V   method (str): a string of either squared_error, friedman_mse, absolute error or poisson that is used as criterion for DecisionTreeRegressor  
        functions:
            I    predict:  A function that uses decision tree to predict the closing price of Bitcoins
        ------------       
        """
        self.depth = depth
        self.predicted_days = predicted_days
        self.data_frame = data_frame
        self.randomnes = randomnes
        self.method = method

    def predict(self):
        """
        Description:
        ------------
            A function that Predict Bitcoin prices and plots the result to the actual price. 
            This function is based on the following code: https://www.kaggle.com/code/rishidamarla/stock-market-prediction-using-decision-tree
        """

        df = pd.DataFrame()
        df = self.data_frame 
        #drop the label colum in datset
        df = df.drop(["Label"], axis=1)
        #Include a prediction colum that is the price of Bitcoin when market close for all days expet the prediction days
        df["Prediction"] = df["Close"].shift(-self.predicted_days)
        #removes the prediction for the prediction days from data set
        X = df.drop(["Prediction"], axis=1)[:-self.predicted_days]
        y = df["Prediction"][:-self.predicted_days]

        #uses the datset to creat a model 
        clf = DecisionTreeRegressor(criterion=self.method, random_state=self.randomnes)
        #Fit model with data tath does not include the predicted days
        clf.fit(X, y)

        #creat data_frem with predicted days
        x_future = df.drop(["Prediction"], axis = 1)[-self.predicted_days:]
        x_future = x_future.tail(self.predicted_days)
        
        # Predict prices of future days
        tree_prediction = clf.predict(x_future)
        predictions = tree_prediction 
        
        #Adds the result for predicted day in a separate data_frame
        valid = x_future
        valid["Prediction"] = predictions

        cwd = os.getcwd()
        path = Path(cwd) / "FigurePlots" / "Decision_tree"
        if not path.exists():
            path.mkdir()

        #Plott result against actuall predicted prices
        lenght = len(X.loc[:,"Close"])
        days = np.linspace(lenght + 1 + 108, lenght + self.predicted_days + 1 + 108, self.predicted_days)

        plt.figure(figsize=(12,8))
        plt.xlabel("Days")
        plt.plot(self.data_frame.loc[:,"Close"], "--", label="Actual closing price")
        plt.scatter(days, valid.loc[:,"Prediction"], label="Predicted closing price", color="tab:orange")
        plt.legend()
        plt.savefig(path / "predicition.png")
        plt.show()
        plt.close()



