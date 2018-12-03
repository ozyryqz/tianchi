import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import pca
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
#读入数据
train_data = pd.read_csv("zhengqi_train.csv")
test_data = pd.read_csv("zhengqi_test.csv")
#划分数据
x_train = np.array(train_data.drop(["target"],axis = 1))
y_train = np.array(train_data.target)
#交叉验证
train_x,test_x,train_y,test_y = train_test_split(x_train,y_train,test_size = 0.2,random_state = 0)
#Bayesian回归
model = BayesianRidge()
model.fit(train_x,train_y)
predict_test = model.predict(test_x)
print(predict_test.shape,test_y.shape)
print("均方误差为：{}".format(mean_squared_error(test_y,predict_test)))

zhengqi = model.predict(test_data)
print(zhengqi.shape)
#print(test_data.shape)
df = pd.DataFrame(zhengqi)
df.to_csv("Bayesian.txt",index=False,header=False)
