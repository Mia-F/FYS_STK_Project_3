import numpy as np
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from os import system
import graphviz 



fontsize = 30
sns.set_theme()
params = {
    "font.family": "Serif",
    "font.serif": "Roman", 
    "text.usetex": True,
    "axes.titlesize": fontsize,
    "axes.labelsize": fontsize,
    "xtick.labelsize": "xx-large",
    "ytick.labelsize": "xx-large",
    "legend.fontsize": fontsize
}
plt.rcParams.update(params)
pd.options.mode.chained_assignment = None  # default='warn'

class Gini_Decision_tree:
    def __init__(self,  X_train, y_train, X_test, y_test, printing=False, data_frame=None, depth=3, predicted_days = 10,) -> None:
        self.depth = depth
        self.predicted_days = predicted_days
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.printing = printing
        self.data_frame = data_frame
    
    def predict(self):
        """
        A function using skit-learns decisonTree with gini index to predict prices
        """
        #self.clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=self.depth, random_state=0)
        self.clf = DecisionTreeRegressor(ccp_alpha=0, max_depth=self.depth, random_state=0)
        # fit the model
        self.clf.fit(self.X_train, self.y_train)
        self.y_pred_gini = self.clf.predict(self.X_test)
        self.y_pred_train_gini = self.clf.predict(self.X_train)
        dotfile = open("dtree.dot", 'w')
        dotfile = tree.export_graphviz(self.clf, out_file = dotfile, feature_names = X.columns)
        system("dot -Tpng dtree.dot -o dtree.png")

        return self.y_pred_gini, self.y_pred_train_gini

    def accuracy(self):
        """
        A function returning the accuracy scores for the training and testing data
        """
        self.accuracy_score_test = accuracy_score(self.y_test, self.y_pred_gini)
        self.accuracy_score_train = accuracy_score(self.y_train, self.y_pred_train_gini)
        if self.printing==True:
            print(f"Model accuracy score with gini index: {self.accuracy_score_test:.4f}")
            print(f"Training-set accuracy score with gini index: {self.accuracy_score_train:.4f}")
        return self.accuracy_score_test, self.accuracy_score_train
    
    def R2_score(self):
        """
        A function returning the R2 score for both the testing and training data
        """
        self.r2_score_test = r2_score(self.y_test, self.y_pred_gini)
        self.r2_score_train = r2_score(self.y_train, self.y_pred_train_gini)
        if self.printing==True:
            print(f"Model R2 score with gini index: {self.r2_score_test:.4f}")
            print(f"Training-set R2 score with gini index: {self.r2_score_train:.4f}")
        return self.r2_score_test, self.r2_score_train
    
    def mse(self):
        """
        A function returning the MSE for both the testing and training data
        """
        self.mse_test = mean_squared_error(self.y_test, self.y_pred_gini)
        self.mse_train = mean_squared_error(self.y_train, self.y_pred_train_gini)
        if self.printing==True:
            print(f"Model MSE value with gini index: {self.mse_test:.4f}")
            print(f"Training-set MSE value with gini index: {self.mse_train:.4f}")
        return self.mse_test, self.mse_train
    
    def predict_future(self):
  
        df = pd.DataFrame()
        df = self.data_frame
        df = df.drop(["Date", "Label"], axis=1)
        df["Prediction"] = df["Close"].shift(-self.predicted_days)

        X = df.drop(["Prediction"], axis=1)[:-self.predicted_days]
        y = df["Prediction"][:-self.predicted_days]

       # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        self.clf.fit(X, y)
        print(X)
        x_future = df.drop(["Prediction"], axis = 1)[-self.predicted_days:]
        print(x_future)
        x_future = x_future.tail(self.predicted_days)
        
        tree_prediction = self.clf.predict(x_future)
        predictions = tree_prediction 
        #print(predictions)
        
        valid = df[X.shape[0]:]
        valid["Prediction"] = predictions
        #print(valid.tail())
        #df["Prediction"] = predictions

        plt.figure(figsize=(16,8))
        plt.title("Model")
        plt.xlabel('Days')
        plt.plot(X.loc[:,"Close"])
        plt.plot(valid.loc[:,"Prediction"])
        plt.plot(self.data_frame.loc[:,"Close"])
        plt.legend(["Original", "Predicted"])
        plt.show()

  

data_frame =pd.read_csv("/Users/miafrivik/Documents/GitHub/FYS_STK_Project_3/Data/BTC-USD_2014.csv")

df = pd.DataFrame(data_frame)
#print(df)

#X = df[df.columns[1:-2]]
X = df.drop(["Date", "Label"], axis=1)
y = data_frame.loc[:, "Target"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8)

Model_gini = Gini_Decision_tree(X_train, y_train, X_test, y_test, printing=True, data_frame=df, depth=100, predicted_days=100)
y_pred_gini, y_pred_train_gini = Model_gini.predict()
#accuracy_score_test, accuracy_score_train = Model_gini.accuracy()
r2_score_test, r2_score_train = Model_gini.R2_score()
mse_test, mse_train = Model_gini.mse()
Model_gini.predict_future()


"""
print(X_train.head())

clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=50, random_state=0)

# fit the model
clf_gini.fit(X_train, y_train)
y_pred_gini = clf_gini.predict(X_test)

print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))

y_pred_train_gini = clf_gini.predict(X_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))

print('Training set score: {:.4f}'.format(clf_gini.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf_gini.score(X_test, y_test)))


dotfile = open("dtree.dot", 'w')
dotfile = tree.export_graphviz(clf_gini, out_file = dotfile, feature_names = X.columns)
system("dot -Tpng dtree.dot -o dtree.png")


clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=50, random_state=0)

clf_en.fit(X_train, y_train)

y_pred_en = clf_en.predict(X_test)

print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test, y_pred_en)))

y_pred_train_en = clf_en.predict(X_train)

print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_en)))

print('Training set score: {:.4f}'.format(clf_en.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf_en.score(X_test, y_test)))

dotfile = open("dtree_entropy.dot", 'w')
dotfile = tree.export_graphviz(clf_en, out_file = dotfile, feature_names = X.columns)
system("dot -Tpng dtree_entropy.dot -o dtree_entropy.png")
"""

#plt.plot(data_frame.loc[:,"btc_market_price"])
#plt.xlabel("Days since 15 of November 2014")
#plt.ylabel("USD")
#plt.show()
