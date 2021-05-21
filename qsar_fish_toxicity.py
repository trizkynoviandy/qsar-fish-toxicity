'''

This project aims to analyze the quantitative structure-activity relationship 
to predict acute aquatic toxicity towards the fish Pimephales promelas (fathead minnow) 
on a set of 908 chemicals. The method used are the Multiple Linear Regression (MLR) 
and the k-Nearest Neighbor (KNN)

Variables used are:

* CIC0
* SM1_Dz(Z)
* GATS1i
* NdsCH
* NdssC
* MLOGP
* Target variables, LC50 \[-LOG(mol/L)]


Dataset source : https://archive.ics.uci.edu/ml/datasets/QSAR+fish+toxicity

'''
# Import Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
plt.style.use('fivethirtyeight')

# Load the dataset
dataset = pd.read_csv('dataset/qsar_fish_toxicity.csv')

# Generate descriptive statistics of the dataset
described = dataset.describe()
print('\nDescriptive Statistics :\n', described)

# Calculate the correlation of each columns with the LC50
correlation = dataset.corr(method = 'pearson')
print('\nCorrelation :\n')
print(correlation)

# Split the dependent and independent variables in the dataset
x = dataset.iloc[:, 0:6]
y = dataset.iloc[:, 6:7]

# Scales the independent variables
scaler = StandardScaler()
x = scaler.fit_transform(x)

# Split the dataset into training and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 42)

# Create the linear model using the MLR
mlr = LinearRegression().fit(x_train,y_train)
mlr_train_pred = mlr.predict(x_train)
mlr_test_pred = mlr.predict(x_test)

# Evaluate the MLR model
mlr_rsq = mlr.score(x_train,y_train)
mlr_test_rsq = mlr.score(x_test,y_test)
mlr_rmse = np.sqrt(mean_squared_error(y_train, mlr_train_pred))
mlr_test_rmse = np.sqrt(mean_squared_error(y_test, mlr_test_pred))

# Create the nonlinear model using the KNN
knn = KNeighborsRegressor().fit(x_train,y_train)
knn_train_pred = knn.predict(x_train)
knn_test_pred = knn.predict(x_test)

# Evaluate the KNN model
knn_rsq = knn.score(x_train,y_train)
knn_test_rsq = knn.score(x_train,y_train)
knn_rmse = np.sqrt(mean_squared_error(y_train, knn_train_pred))
knn_test_rmse = np.sqrt(mean_squared_error(y_test, knn_test_pred))

# Model evaluation using theR2 and the RMSE metrics
print('\nMLR R-Squared : {:.3f} for the training set, and {:.3f} for the testing set'.format(mlr_rsq, mlr_test_rsq))
print('MLR RMSE : {:.3f} for the training set, and {:.3f} for the testing set\n'.format(mlr_rmse, mlr_test_rmse))
print('KNN R-Squared : {:.3f} for the training set, and {:.3f} for the testing set'.format(knn_rsq, knn_test_rsq))
print('KNN RMSE : {:.3f} for the training set, and {:.3f} for the testing set'.format(knn_rmse, knn_test_rmse))

# Visualize the regression plot and the residual plot of the KNN method
residual_train = y_train - knn_train_pred
residual_test = y_test - knn_test_pred

fig=plt.figure(figsize=(15,5))

ax1=plt.subplot(1,2,1)
ax1.scatter(y_train, knn_train_pred, s=15, label='training')
ax1.scatter(y_test, knn_test_pred, marker='^', s=15, c='r', label='testing')
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'k', lw=1)
ax1.set_title('Prediction Plot')
ax1.set_xlabel('Observation')
ax1.set_ylabel('Prediction')
ax1.legend()

ax2=plt.subplot(1,2,2)
ax2.plot(y_train, residual_train, 'o', markersize= 4, label='training')
ax2.plot(y_test, residual_test, '^', c='red', markersize= 4, label='testing')
ax2.axhline(y=0.3, linewidth= 1, linestyle='-', c='black')
ax2.legend()
ax2.set_title('Residual Plot')
ax2.set_xlabel('Target')
ax2.set_ylabel('Residual')

plt.show()