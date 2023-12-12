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

random.seed(10)
class Decision_tree:
    def __init__(self,  X_train, y_train, X_test, y_test, printing=False, depth=3, randomnes = 0, alpha= False, method="squared_error") -> None:
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
        A function using skit-learns decisonTreeRegressor to fit the model to the dataset 
        """
        clf = DecisionTreeRegressor(criterion=self.method, max_depth=self.depth, random_state=self.randomnes, ccp_alpha=self.alpha)
        if self.alpha == True:
            
            #Define path for saving plots
            cwd = os.getcwd()
            
            path = Path(cwd) / "FigurePlots" / "Decision_tree"/ self.method 
            if not path.exists():
                path.mkdir()

            path_pruning = clf.cost_complexity_pruning_path(self.X_train, self.y_train)
            ccp_alphas, impurities = path_pruning.ccp_alphas, path_pruning.impurities
            clfs = []

            fig, ax = plt.subplots()
            ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
            ax.set_xlabel("effective alpha")
            ax.set_ylabel("total impurity of leaves")
            #ax.set_title("Total Impurity vs effective alpha for training set")
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
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
            ax[0].set_xlabel("alpha")
            ax[0].set_ylabel("number of nodes")
            #ax[0].set_title("Number of nodes vs alpha")
            ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
            ax[1].set_xlabel("alpha")
            ax[1].set_ylabel("depth of tree")
            #ax[1].set_title("Depth vs alpha")
            plt.savefig(path / "Alpha_depth.png")
            plt.close()

            fig, ax = plt.subplots()
            ax.plot(ccp_alphas, mse[:-1], label="Test")
            ax.plot(ccp_alphas, mse_train[:-1], label="Train")
            ax.set_xlabel("alpha")
            ax.set_ylabel("MSE")
            #ax.set_title("MSE vs alpha")
            ax.legend()
            plt.savefig(path / "Alpha_mse.png")
            plt.close()
                    
        else:   
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            y_pred_train = clf.predict(self.X_train)
            return y_pred, y_pred_train
        
    

    def R2_score(self, y_pred, y_pred_train):
        """
        A function returning the R2 score for both the testing and training data
        """
        r2_score_test = r2_score(self.y_test, y_pred)
        r2_score_train = r2_score(self.y_train, y_pred_train)
        if self.printing==True:
            print(f"Model R2 score: {r2_score_test:.4f}")
            print(f"Training-set R2 score: {r2_score_train:.4f}")
        return r2_score_test, r2_score_train
    
    def mse(self, y_pred, y_pred_train):
        """
        A function returning the MSE for both the testing and training data
        """
        mse_test = mean_squared_error(self.y_test, y_pred)
        mse_train = mean_squared_error(self.y_train, y_pred_train)
        if self.printing==True:
            print(f"Model MSE value: {mse_test:.4f}")
            print(f"Training-set MSE value: {mse_train:.4f}")
        return mse_test, mse_train
    
class predict_future_tree:
    def __init__(self, X,y , printing=False, data_frame=None, depth=3, predicted_days = 10, randomnes = 0, method="squared_error") -> None:
        self.depth = depth
        self.predicted_days = predicted_days
        self.X = X
        self.y = y
        self.printing = printing
        self.data_frame = data_frame
        self.randomnes = randomnes
        self.method = method

    def predict(self):

        df = pd.DataFrame()
        df = self.data_frame 
        df = df.drop(["Label"], axis=1)
        df["Prediction"] = df["Close"].shift(-self.predicted_days)

        X = df.drop(["Prediction"], axis=1)[:-self.predicted_days]
        print(X)
        y = df["Prediction"][:-self.predicted_days]

       
        clf = DecisionTreeRegressor(criterion=self.method, random_state=self.randomnes)
        clf.fit(X, y)
        x_future = df.drop(["Prediction"], axis = 1)[-self.predicted_days:]
        print(x_future)
        x_future = x_future.tail(self.predicted_days)
        
        tree_prediction = clf.predict(x_future)
        predictions = tree_prediction 
        #print(predictions)
        
        valid = x_future
        valid["Prediction"] = predictions
        print(valid.tail())
        #print(valid.tail())
        #df["Prediction"] = predictions
        cwd = os.getcwd()
        path = Path(cwd) / "FigurePlots" / "Decision_tree"
        if not path.exists():
            path.mkdir()
        plt.figure(figsize=(16,8))
        plt.title("Model")
        plt.xlabel('Days')
        plt.plot(X.loc[:,"Close"])
        plt.plot(valid.loc[:,"Prediction"])
        #plt.plot(self.data_frame.loc[:,"Close"])
        plt.legend(["Original", "Predicted"])
        plt.savefig(path / "predicition.png")
        plt.close()



