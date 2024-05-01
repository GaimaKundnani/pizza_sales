import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
#Loading Dataset
df=pd.read_csv(r'C:\Users\garim\OneDrive\Desktop\Data Science\pizza_sales.csv')
#Infornation of Dataset
print(df.head())
print("Information:")
print(df.info())
print("Description:")
print(df.describe())
#Sorting Values
print("Sort Values:")
df2=df.sort_values(by='quantity')
print(df2.head())
# Logistic Regression
X_train,X_test,y_train,y_test=train_test_split(df2[['quantity']],df[['total_price']],test_size=0.2,random_state=15)
model=LinearRegression()
model.fit(X_train,y_train)
# Convert y_train and y_test to 1d arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()
#SVM
svr_model = SVR(kernel='rbf', C=100, gamma='auto')
svr_model.fit(X_train, y_train)
y_pred = svr_model.predict(X_test)
print(svr_model.score(X_test,y_test)*100)

#Gaussian Process Regression
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
#gpr_model = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=10, random_state=15)
# Create Sparse Gaussian Process Regression model
gpr_model = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=10, random_state=15, optimizer='fmin_l_bfgs_b', normalize_y=True)
gpr_model.fit(X_train, y_train)
y_pred, std_dev = gpr_model.predict(X_test, return_std=True)
print(gpr_model.score(X_test,y_test)*100)
