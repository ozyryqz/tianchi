#直接对数据使用pca降维，然后进行线性回归
import numpy as np 
import pandas as pd 
from sklearn.linear_model import BayesianRidge 
from sklearn.decomposition import pca
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
#读入数据
train_data = pd.read_csv("zhengqi_train.csv")
test_data = pd.read_csv("zhengqi_test.csv")
#进行相关性分析
column = train_data.columns.tolist()
mcorr = train_data[column].corr()
print(mcorr.target)

x_trdata = train_data.drop(["V14","V21","V25","V26","V28","V32","V33","V34"],axis = 1)
#print(x_data.columns.tolist())
x_tedata = test_data.drop(["V14","V21","V25","V26","V28","V32","V33","V34"],axis = 1)
x_tr = x_trdata.drop(["target"],axis = 1)
y_tr = x_trdata.target
#进行pca
pca = pca.PCA(n_components=0.95)
pca.fit(x_tr)
x_tr_pca = pca.transform(x_tr)
x_te_pca = pca.transform(x_tedata)
#划分测试集
x_train,x_test,y_train,y_test = train_test_split(x_tr_pca,y_tr,test_size = 0.2,random_state=0)

model = BayesianRidge()
model.fit(x_train,y_train)
a = model.intercept_
b = model.coef_
#print(a,b)

pre_y = model.predict(x_test)
error = mean_squared_error(pre_y,y_test)
print("在pca下线性回归的误差为：{}".format(error))

zhengqi = model.predict(x_te_pca)
print(zhengqi.shape)
#print(test_data.shape)
df = pd.DataFrame(zhengqi)
df.to_csv("Bayesian_feature_drop_pca1.txt",index=False,header=False)