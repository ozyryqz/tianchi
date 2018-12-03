import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import pca
#读数据
train_data = pd.read_csv("zhengqi_train.csv")
test_data = pd.read_csv("zhengqi_test.csv")
data = test_data.describe()
print(data)
d = pd.DataFrame(data)
print("hyuguyuiy",d.std)
#相关性分析
mcorr = train_data[train_data.columns.tolist()[:39]].corr()
print(np.array(mcorr.target))
x = pd.DataFrame(train_data.drop(["V14","V21","V25","V26","V28","V32","V33","V34"],axis=1))
Mcorr = x[x.columns.tolist()].corr()
Mcorr_data = np.array(Mcorr.target)
print("Mcorr各变量同目标之间的相关系数为：{}".format(Mcorr_data))
#对数据进行划分
x_train = np.array(x.drop(["target"],axis = 1))
y_train = np.array(x.target)
x_test = pd.DataFrame(test_data.drop({"V14","V21","V25","V26","V28","V32","V33","V34"},axis=1))
#PCA过程
pca = pca.PCA(n_components=0.95) #0.95
pca.fit(x_train)
train_pca = pca.transform(x_train)
test_pca = pca.transform(x_test)
#划分训练集进行线性回归
train_x,test_x,train_y,test_y = train_test_split(train_pca,y_train,test_size = 0.2,random_state = 0)

model = BayesianRidge()
model.fit(train_x,train_y)
a = model.intercept_
b = model.coef_
print("回归系数为：{}".format(b))
print("截距是：{}".format(a))
predict_test = model.predict(test_x)
print(predict_test.shape)
print(test_y.shape)
print("测试数据和预测数据{}，{}".format(y_train.shape,predict_test.shape))
print(mean_squared_error(test_y , predict_test))
zhengqi = model.predict(test_pca)
print(zhengqi.shape)
#print(test_data.shape)
df = pd.DataFrame(zhengqi)
df.to_csv("Bayesian_feature_drop_pca.txt",index=False,header=False)