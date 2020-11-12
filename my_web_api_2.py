import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # not sure if we need this in this code
from sklearn.model_selection import GridSearchCV
import time
import numpy.matlib
from sklearn import metrics

# matplotlib inline
bankdata = pd.read_excel(r"/Users/a080528/Desktop/專題/AI/AI/結果/產期/小白菜/bai02.csv",nrow=1700)
print(bankdata.shape)
print(bankdata.head())

# drop rows with nan or inf
# bankdata=bankdata.drop(['PMPlantInfoID','Code','TAFTCode','PMCropName','PMCropTypeName','PlantDate','ID','CountyName',\
 #   'TownshipName','stationname','stationid','採收日期','PerHarvest'],axis=1)
bankdata=bankdata.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

print(bankdata.shape)
print(bankdata.head())
# assign the input (X) and output (y)
X=bankdata.drop(['Days'],axis=1)
y=bankdata['Days']


[X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size = 0.20)
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1),
                   param_grid={"C": [0.01, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-100, 10, 100)})


t0 = time.time()
svr.fit(X_train, y_train)
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s"
      % svr_fit)

t0 = time.time()
y_svr = svr.predict(X_test)
svr_predict = time.time() - t0
print("SVR prediction for %d inputs in %.3f s"
      % (X_test.shape[0], svr_predict))
      
      
the_mape=np.mean(abs(y_test-y_svr)/y_test)
print("MAPE is "+"{:.2%}".format(the_mape))

y_test_mean=np.matlib.repmat(np.mean(y_test),y_test.size,1)

# try_1=(y_test-y_test_mean)
# size_1=y_test_mean.size
# add one more dim to the y_test (e.g., (30,) to (30,1))
y_test=y_test[:,np.newaxis]
y_svr=y_svr[:,np.newaxis]

R2_error=1-(np.sum((y_test-y_svr)**2)/np.sum((y_test-y_test_mean)**2))

print("R2 is %.3f" %R2_error)

# R2_error_2=(np.sum((y_svr-y_test_mean)**2)/np.sum((y_test-y_test_mean)**2))

# print("R2_2 is %.3f" %R2_error_2)


print("R^2 : ", metrics.r2_score(y_test, y_svr))
print("Mean Absolute Error: ", metrics.mean_absolute_error(y_test, y_svr))
print("Mean Squared Error: ", metrics.mean_squared_error(y_test, y_svr))
print("Root Mean Squared Error : ", np.sqrt(metrics.mean_squared_error(y_test, y_svr)))

import pickle
pickle.dump(svr, open(r"/Users/a080528/Desktop/專題/AI/AI/結果/產期/小白菜/final_prediction_bai.pickle", 'wb'))

model_columns = list(X.columns)
print(X.columns)
pickle.dump(model_columns, open(r"/Users/a080528/Desktop/專題/AI/AI/結果/產期/小白菜/model_columns_bai.pickle", 'wb'))
