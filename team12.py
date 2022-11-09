import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from six import StringIO
from sklearn.tree import export_graphviz
from sklearn import tree
import pydotplus
import graphviz
from sklearn.metrics import accuracy_score


# Reading the file

xls = pd.ExcelFile("ebayAuctions.xlsx")
df = pd.read_excel(xls, 'eBay auctions')

# Basic EDA before data-processing

print("The initial shape of the dataframe before any pre-processing is ", df.shape)
print("Does the data contain any null values? ", df.isnull().values.any())
print("Our target variable is 'Competitive?', the value counts for the same are as follows -")
print(df['Competitive?'].value_counts())

# Data pre-processing

df['Duration'] = df['Duration'].astype('category')

df_dumies = pd.get_dummies(df)

print("After converting all the categorical columns to dummy variables, the shape of our dataframe is ", df_dumies.shape)

# Splitting data into training and testing data

X = df_dumies.drop(columns=['Competitive?'])
y = df_dumies['Competitive?']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1)

# Fitting the first tree with all predictors with no restrictions on the number of nodes in the leaf

clf = DecisionTreeClassifier(random_state=1)
clf = clf.fit(X_train, y_train)

# Fitting the first tree with all predictors with a limit of 50 to the number of nodes in the leaf

clf50 = DecisionTreeClassifier(random_state=1, min_samples_split=50)
clf50 = clf50.fit(X_train, y_train)
X_cols = list(X.columns.values)

# Getting the dot_data the decision tree with no limit at the leaf node

export_graphviz(clf, out_file='dot_files/fullClassTree.dot',
                feature_names=X_train.columns)

# Plotting the decision tree with a limit of 50 leaf nodes

export_graphviz(clf50, out_file='dot_files/treewith50nodes.dot',
                feature_names=X_train.columns)

# Measuring Accuracy

y_predicted = clf.predict(X_test)
y_predicted_50 = clf50.predict(X_test)
print("The test accuracy for the decision tree with no restrictions is ")
print((accuracy_score(y_test, y_predicted)))
print("The test accuracy for the decision tree with the restriction of 50 nodes at the leaf is ")
print((accuracy_score(y_test, y_predicted_50)))


# Reducing the predictors

# Since we do not have a way of knowing what is the selling price of the product and the duration of the auction, it does not make
# sense to include them in predicting whether or not an auction will be competitive or not. Therefore we should drop these variables

df = df.drop(columns=['endDay'])
df_dumies = pd.get_dummies(df)
X = df_dumies.drop(
    columns=['Competitive?', 'ClosePrice'])
y = df_dumies['Competitive?']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1)

clf_3 = DecisionTreeClassifier(random_state=1)
clf_3 = clf_3.fit(X_train, y_train)


export_graphviz(clf_3, out_file='dot_files/thirdclassifier.dot',
                feature_names=X_train.columns)

y_predicted_3 = clf_3.predict(X_test)
print("The test accuracy for the decision tree with no restrictions is ")
print((accuracy_score(y_test, y_predicted_3)))
