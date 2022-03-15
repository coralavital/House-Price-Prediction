"""
===
Final Project - AI
===
   Name            ID
Coral Avital    205871163 
Yoni Ifrah      313914723
"""

#===========================================================================
# imports
import sklearn
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import hvplot.pandas



#===========================================================================
print(f"sklearn version: {sklearn.__version__}")

#===========================================================================
#import dataframe as data
data = pd.read_csv('Train.csv')
data.head()

data.info()

data.describe()

data.columns

plt.hist(data["Price"])
plt.xlabel('Price', fontweight ="bold")
plt.show()

plt.scatter(x=data['Avg. Area House Age'], y=data['Price'])
plt.ylabel('Price', fontweight ="bold")
plt.xlabel('Avg. Area House Age', fontweight ="bold")
plt.show()

plt.scatter(x=data['Avg. Area Income'], y=data['Price'])
plt.xlabel('Avg. Area Income', fontweight ="bold")
plt.ylabel('Price', fontweight ="bold")
plt.show()

plt.scatter(x=data['Avg. Area Number of Rooms'], y=data['Price'])
plt.xlabel('Avg. Area Number of Rooms', fontweight ="bold")
plt.ylabel('Price', fontweight ="bold")
plt.show()

plt.hist(x=data['Avg. Area Number of Bedrooms'])
plt.xlabel('Avg. Area Number of Bedrooms', fontweight ="bold")
plt.ylabel('Amount', fontweight ="bold")
plt.show()

plt.scatter(x=data['Area Population'], y=data['Price'])
plt.xlabel('Area Population', fontweight ="bold")
plt.ylabel('Price', fontweight ="bold")
plt.show()

sns.heatmap(data.corr(), annot=True)
plt.show()

X = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def cross_val(model):
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

pipeline = Pipeline([
    ('std_scalar', StandardScaler())
])

X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)

#===========================================================================
# Model no 1.
#Linear Regression
def create_Linear_Regrresion_model():
    lin_reg = LinearRegression()
    lin_reg.fit(X_train,y_train)
    print('\nModel number 1 - Linear Regression')
    coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])
    print(coeff_df)
    pred = lin_reg.predict(X_test)
    pd.DataFrame({'True Values': y_test, 'Predicted Values': pred}).hvplot.scatter(x='True Values', y='Predicted Values')
    pd.DataFrame({'Error Values': (y_test - pred)}).hvplot.kde()
    test_pred = lin_reg.predict(X_test)
    train_pred = lin_reg.predict(X_train)
    print('Test set evaluation:\n_____________________________________')
    print_evaluate(y_test, test_pred)
    print('Train set evaluation:\n_____________________________________')
    print_evaluate(y_train, train_pred)
    results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, test_pred) , cross_val(LinearRegression())]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
    print(results_df)
    plt.figure(figsize=(8, 8))
    sns.regplot(y_test, test_pred, scatter_kws=dict(color="#7583EA"), line_kws=dict(color="#9EA2C1", linewidth=3))
    plt.title("Linear Regression Actual vs Predict Train Data")
    plt.xlabel("Actual Value")
    plt.ylabel("Predict Value")
    plt.show()
    
create_Linear_Regrresion_model()

#===========================================================================
# Model no 2.
# KNN
def create_KNN_model():   
    n_neighbors = [3, 8, 10]
    for i in n_neighbors:
        print('\nModel number 2 - KNN with {} neighbors'.format(i))
        knn = KNeighborsRegressor(i)
        knn.fit(X_train,y_train)    
        coeff_df = pd.DataFrame(knn.score(X=X_train, y=y_train), X.columns, columns=['Coefficient'])
        print(coeff_df)
        test_pred = knn.predict(X_test)
        train_pred = knn.predict(X_train)
        print('Test set evaluation:\n_____________________________________')
        print_evaluate(y_test, test_pred)
        print('Train set evaluation:\n_____________________________________')
        print_evaluate(y_train, train_pred)
        plt.figure(figsize=(8, 8))
        sns.regplot(y_test, test_pred, scatter_kws=dict(color="#7583EA"), line_kws=dict(color="#9EA2C1", linewidth=3))
        plt.title("KNN with {} neighbors Actual vs Predict Train Data".format(i))
        plt.xlabel("Actual Value")
        plt.ylabel("Predict Value")
        plt.show()

create_KNN_model()

# #===========================================================================
#find HyperParameters For Knn
def find_hyperparameters_knn():
    #List Hyperparameters that we want to tune.
    leaf_size = list(range(1, 50))
    n_neighbors = list(range(1, 20))
    p = [1, 2]
    #Convert to dictionary
    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
    #Create new KNN object
    knn_2 = knn = KNeighborsRegressor()
    #Use GridSearch
    knn = GridSearchCV(knn_2, hyperparameters, cv=10)
    #Fit the model
    best_model = knn.fit(X_train, y_train)
    #Print The value of best Hyperparameters
    print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])
    return best_model

#Knn With Hyperparameters
def knn_with_hyperparameters(knn):
    coeff_df = pd.DataFrame(knn.score(X=X_train,y=y_train), X.columns, columns=['Coefficient'])
    print(coeff_df)
    test_pred = knn.predict(X_test)
    train_pred = knn.predict(X_train)
    print('\nModel number 2 - KNN with Hyperparameters')
    print('Test set evaluation:\n_____________________________________')
    print_evaluate(y_test, test_pred)
    print('Train set evaluation:\n_____________________________________')
    print_evaluate(y_train, train_pred)
    plt.figure(figsize=(8, 8))
    sns.regplot(y_test, test_pred, scatter_kws=dict(color="#7583EA"), line_kws=dict(color="#9EA2C1", linewidth=3))
    plt.title("KNN with Hyperparameters Actual vs Predict Train Data")
    plt.xlabel("Actual Value")
    plt.ylabel("Predict Value")
    plt.show()

knn_with_hyperparameters(find_hyperparameters_knn())