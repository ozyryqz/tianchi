#直接对数据使用pca降维，然后进行线性回归
import numpy as np 
import pandas as pd 
from sklearn.linear_model import LinearRegression 
from sklearn.decomposition import pca
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
#读入数据
train_data = pd.read_csv("zhengqi_train.csv")
test_data = pd.read_csv("zhengqi_test.csv")
x_tr = train_data.drop(["target"],axis = 1)
y_tr = np.array(train_data.target)

#使用pca
pca = pca.PCA(n_components=0.95)
pca.fit(x_tr)
x_tr_pca = pca.transform(x_tr)
x_te_pca = pca.transform(test_data)

#划分测试集
x_train,x_test,y_train,y_test = train_test_split(x_tr_pca,y_tr,test_size = 0.2,random_state=0)

model = LinearRegression()
model.fit(x_train,y_train)
a = model.intercept_
b = model.coef_
#print(a,b)

pre_y = model.predict(x_test)
error = mean_squared_error(pre_y,y_test)
print("在pca下线性回归的误差为：{}".format(error))
