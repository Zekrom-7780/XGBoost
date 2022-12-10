from tkinter import Y
import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed, Conv1D, MaxPooling1D, Flatten, Bidirectional, Input, Flatten, Activation, Reshape, RepeatVector, Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  
x = pd.read_csv('D:\GUIs (5)\XGBoost\XGBoost\data_14245.csv')
x.shape
x.head()
import pandas as pd # for data manipulation
import numpy as np # for data manipulation
from sklearn.linear_model import LinearRegression # for building a linear regression model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split,RandomizedSearchCV,GridSearchCV,ShuffleSplit
from datetime import datetime
import tensorflow as tf
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
z=x.iloc[:,-1]
z.head()
print(z.shape)
y=x.drop(['Q'],axis=1)
y.head()
print(y.shape)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler
columns=y.columns
# mean = y.mean(axis=0)
# print(mean.shape)
# std = y.std(axis=0)
# print(std.shape)
# train_data = (y - mean) / std

# mean1 = z.mean(axis=0)
# print(mean1.shape)
# std1 = z.std(axis=0)
# print(std1.shape)
# test_data = (z-mean1) /std1

# print(train_data.shape)
# print(test_data.shape)

import numpy as np
import pandas as pd

# def quantileNormalize(df_input):
#     df = df_input.copy()
#     #compute rank
#     dic = {}
#     for col in df:
#         dic.update({col : sorted(df[col])})
#     sorted_df = pd.DataFrame(dic)
#     rank = sorted_df.mean(axis = 1).tolist()
#     #sort
#     for col in df:
#         t = np.searchsorted(np.sort(df[col]), df[col])
#         df[col] = [rank[i] for i in t]
#     return df

train_data=y
print(train_data.shape)
z=pd.DataFrame(z)
test_data=z
print(z.shape)
print(test_data.shape)

X_train,X_test,Y_train,Y_test=train_test_split(train_data,test_data,test_size=0.2,random_state=123)
sample_size = X_train.shape[0] # number of samples in train set
time_steps  = X_train.shape[1] # number of features in train set
input_dimension = 1 
X_train=np.array(X_train)
train_data_reshaped = X_train.reshape(sample_size,time_steps,input_dimension)
print("After reshape train data set shape:\n", train_data_reshaped.shape)
print("1 Sample shape:\n",train_data_reshaped[0].shape)
print("An example sample:\n", train_data_reshaped[0])

sample_size1 = X_test.shape[0] # number of samples in train set
time_steps1  = X_test.shape[1] # number of features in train set
input_dimension1 = 1 
X_test=np.array(X_test)
test_data_reshaped = X_test.reshape(sample_size1,time_steps1,input_dimension1)
print("After reshape train data set shape:\n", test_data_reshaped.shape)
print("1 Sample shape:\n",test_data_reshaped[0].shape)
print("An example sample:\n", test_data_reshaped[0])

def build_conv1D_model():

    n_timesteps = train_data_reshaped.shape[1] #13
    n_features  = train_data_reshaped.shape[2] #1 
    model = keras.Sequential()
    # model.add(Dense(shape=))
    model.add(Conv1D(filters=128, kernel_size=15, activation='relu', name="Conv1D_1",input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=128, kernel_size=15, activation='relu', name="Conv1D_2"))

    model.add(Conv1D(filters=16, kernel_size=15, activation='relu', name="Conv1D_3"))

    model.add(MaxPooling1D(pool_size=4, name="MaxPooling1D"))
    model.add(Flatten())
    model.add(Dense(32, activation='relu', name="Dense_1"))
    model.add(Dense(n_features, name="Dense_2"))


    optimizer = tf.keras.optimizers.Adam(0.001)

    model.compile(loss='mse',optimizer=optimizer,metrics=['mae'])
    return model

model_conv1D = build_conv1D_model()
model_conv1D.summary()
EPOCHS=5
history = model_conv1D.fit(train_data_reshaped, Y_train, epochs=EPOCHS,
                    validation_split=0.2, verbose=1,batch_size=128)

y_pred=model_conv1D.predict(test_data_reshaped)

r2_score=r2_score(Y_test,y_pred)