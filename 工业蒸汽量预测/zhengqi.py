import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas import DataFrame,Series
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
#读入数据集
data = pd.read_csv("zhengqi_train.csv")
test = pd.read_csv("zhengqi_test.csv")
#清洗数据
#new_data = data.ix[:1:]
#查看数据集前几行以及其形状
print(data.head(5))
print("This data set is:{}".format(data.shape))
#数据描述
#print(data.describe())
#缺失值检验
#print(data[data.isnull()==True].count())
#绘制盒图
data.boxplot()
plt.savefig("boxplot.png")
plt.xticks(rotation=90)
plt.show()
##相关系数矩阵 r(相关系数) = x和y的协方差/(x的标准差*y的标准差) == cov（x,y）/σx*σy
#相关系数0~0.3弱相关0.3~0.6中等程度相关0.6~1强相关
#print(data.corr())

#sns.pairplot(data,x_vars=["V0","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","V29","V30","V31","V32","V33","V34","V35","V36","V37"],y_vars="target", palette="husl",kind="reg")
#plt.show()
X_train,X_test,Y_train,Y_test = train_test_split(data.ix[:,:38],data.target,train_size=.80)
 
print("原始数据特征:",data.ix[:38].shape,
      ",训练数据特征:",X_train.shape,
      ",测试数据特征:",X_test.shape)
 
print("原始数据标签:",data.target.shape,
      ",训练数据标签:",Y_train.shape)
model = LinearRegression()
 
model.fit(X_train,Y_train)
 
a  = model.intercept_#截距
 
b = model.coef_#回归系数
 
print("最佳拟合线:截距",a,",回归系数：",b)
#R方检测
#决定系数r平方
#对于评估模型的精确度
#y误差平方和 = Σ(y实际值 - y预测值)^2
#y的总波动 = Σ(y实际值 - y平均值)^2
#有多少百分比的y波动没有被回归拟合线所描述 = SSE/总波动
#有多少百分比的y波动被回归线描述 = 1 - SSE/总波动 = 决定系数R平方
#对于决定系数R平方来说1） 回归线拟合程度：有多少百分比的y波动刻印有回归线来描述(x的波动变化)
#2）值大小：R平方越高，回归模型越精确(取值范围0~1)，1无误差，0无法完成拟合
#score = model.score(X_test,Y_test)
#print(score)
#对线性回归进行预测
Y_pred = model.predict(test)
print(Y_pred)
print(type(Y_pred))
df = pd.DataFrame(Y_pred)
df.to_csv('1.txt',index=False,header=False)
#plt.plot(range(len(Y_pred)),Y_pred,"b",label="predict")
#plt.show()