import numpy as np
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
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



data_frame =pd.read_csv("/Users/miafrivik/Documents/GitHub/FYS_STK_Project_3/Data/BTC-USD_2014.csv")

df = pd.DataFrame(data_frame)


X = df[df.columns[1:-2]]

print(np.shape(X))

y = data_frame.loc[:, "Label"]



#plt.plot(data_frame.loc[:,"btc_market_price"])
#plt.xlabel("Days since 15 of November 2014")
#plt.ylabel("USD")
#plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8)

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


print(clf_gini)
print(X_train.columns)
print(y_train)

dotfile = open("dtree.dot", 'w')
dotfile = tree.export_graphviz(clf_gini, out_file = dotfile, feature_names = X.columns)
system("dot -Tpng dtree.dot -o dtree.png")
