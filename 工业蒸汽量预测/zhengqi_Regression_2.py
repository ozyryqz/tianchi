#对数据进行相关性分析剔除correct小于0.1的数据后进行线性回归
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#读入数据
train_data = pd.read_csv("zhengqi_train.csv")
test_data = pd.read_csv("zhengqi_test.csv")

#进行相关性分析
column = train_data.columns.tolist()
mcorr = train_data[column].corr()
print(mcorr.target)

x_data = train_data.drop(["V14","V21","V25","V26","V28","V32","V33","V34"],axis = 1)
#print(x_data.columns.tolist())

#划分数据集
x = x_data.drop(["target"],axis = 1)
y = x_data.target
#print(x.shape,y.shape)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

#进行线性回归
model = LinearRegression()
model.fit(x_train,y_train)
pre_y = model.predict(x_test)
#print(pre_y.shape)
error = mean_squared_error(pre_y,y_test)
print("在剔除相关性比较差的特征后使用线性回归的均方误差为：{}".format(error))