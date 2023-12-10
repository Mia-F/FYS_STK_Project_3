import numpy as np
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree

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

clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)

# fit the model
clf_gini.fit(X_train, y_train)
y_pred_train_en = clf_en.predict(X_train)

plt.figure(figsize=(12,8))
tree.plot_tree(clf_gini.fit(X_train, y_train)) 
plt.show()


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_en)

print('Confusion matrix\n\n', cm)