#!/bin/python3

import math
import os
import random
import re
import sys
import pandas as pd



if __name__ == '__main__':
    timeCharged = float(input().strip())
    
    data_dict = {}

    with open("trainingdata.txt", "r") as file:
        for line in file:
            key, value = line.strip().split(",")
            pair = {key:value}
            data_dict.update(pair)
            
    data_dict = {float(key): float(value) for key, value in data_dict.items()}
    data_dict


    # In[96]:


    # converting dictionary to pandas dataframe

    df = pd.DataFrame(list(data_dict.items()), columns=["Charged time", "Battery time"])
    df.head(5)


    # In[97]:


    x = df[['Charged time']][:]
    y = df[['Battery time']][:]

    type(x)


    # In[98]:


    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)


    # In[99]:


    # Assuming x_train and y_train are Pandas DataFrames or Series

    # Convert y_train to Series if it's a NumPy array
    y_series = y_train.squeeze()

    # Create a boolean mask where y != 8.0
    mask = y_series != 8.0

    # Apply the mask to both x and y
    x_clean = x_train[mask]
    y_clean = y_series[mask]


    # In[100]:


    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(x_clean, y_clean)


    # In[101]:


    def get_predictions(input=None):
        threshold = 4
        if isinstance(input, float):
            if input < threshold:
                preds = model.predict(pd.DataFrame([[input]], columns=['Charged time']))
                return preds[0]
            else:
                return 8.0
        else:
            preds_arr = []
            for data in input['Charged time']:
                if data < threshold:
                    prd = model.predict(pd.DataFrame([[data]], columns=['Charged time']))
                    preds_arr.append(prd[0])
                else:
                    preds_arr.append(8.0)
            return pd.DataFrame(preds_arr, columns=['Predictions'])


    # In[102]:


    y_pred = get_predictions(x_train)   # Makes predictions

    y_pred[:10]


    # In[103]:


    # plt.figure(figsize=(10, 7))
    # plt.xlabel("Charged Time")
    # plt.ylabel("Battery Time")

    # plt.scatter(x=x_train, y=y_train, c='r', s=4, label="Training data")
    # plt.scatter(x=x_train, y=y_pred, c='g', s=4, label="Testing data")


    # plt.legend(prop={"size": 14})

    # plt.show()


    # In[104]:

    pred = get_predictions(timeCharged)
    print(pred)

