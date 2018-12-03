#不对数据进行任何处理直接进行LinearRegression
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#读数据
train_data = pd.read_csv("zhengqi_train.csv")
test_data = pd.read_csv("zhengqi_test.csv")

#对数据集进行划分
x_trdata = train_data.drop(["target"],axis = 1)
print(type(x_trdata))
y_trdata = train_data.target
x_train,x_test,y_train,y_test = train_test_split(x_trdata,y_trdata,test_size = 0.2,random_state = 0)

#进行线性回归
model = LinearRegression()
model.fit(x_train,y_train)
predit_tr = model.predict(x_test)
error = mean_squared_error(predit_tr,y_test)
print("该模型线性均方误差为：{}".format(error))

zhengqi = model.predict(test_data)
print(zhengqi.shape)
#print(test_data.shape)
df = pd.DataFrame(zhengqi)
df.to_csv("LinearRegression.txt",index=False,header=False)