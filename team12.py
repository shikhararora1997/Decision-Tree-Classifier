import pandas as pd
from sklearn.model_selection import train_test_split

# Reading the file

xls = pd.ExcelFile("ebayAuctions.xlsx")
df = pd.read_excel(xls, 'eBay auctions')

# Basic EDA before data-processing

print("The initial shape of the dataframe before any pre-processing is ", df.shape)
print("Does the data contain any null values? ", df.isnull().values.any())
print("Our target variable is 'Competitive?', the value counts for the same are as follows -")
print(df['Competitive?'].value_counts())

# Data pre-processing

df_dumies = pd.get_dummies(df)
print("After converting all the categorical columns to dummy variables, the shape of our dataframe is ", df_dumies.shape)

# Splitting data into training and testing data

X = df_dumies.drop(columns=['Competitive?'])
y = df_dumies['Competitive?']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1)


# Decision trees are a non-paramatric supervised learning method and require a pre-classified target variable. A training data must be supplied
# which provides the algorithm with the values of the target variable. Therefore it's a great sign that our data does not have any missing data
# We will be using C4.5 as our algorithm because unlike the CART algorithm C4.5 is not limited to just binary splits
