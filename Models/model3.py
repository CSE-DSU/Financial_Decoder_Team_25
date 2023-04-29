import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn import metrics
# loading the csv data to a pandas DataFrame
gold_data=pd.read_csv('/content/gld_price_data.csv')
# print first rows
gold_data.head()
# construct heat map to understand coorlation
plt.figure(figsize=(8,8))
sns.heatmap(correlation,cbar=True,square=True,fmt='.1f',annot=True,
            annot_kws={'size':8},cmap='Blues')
#fmt-no.of decimal pts
# annot-name of col and annot_kws size
regressor = RandomForestRegressor(n_estimators=100)
# Training the model
regressor.fit(X_train,Y_train)
# prediction on test data
test_data_prediction = regressor.predict(X_test)
from scipy.ndimage.measurements import label
plt.plot(Y_test,color='blue',label = 'Actual Value')
plt.plot(test_data_prediction,color='green',label='PredictedValue')
plt.title('Actual Price vs Pred Price')
plt.xlabel("Number of values")
plt.ylabel('GLD Price')
plt.legend()
plt.show()
