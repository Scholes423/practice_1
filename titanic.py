# 导入库
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


# 编写可视化特征函数
def show_feature_importance(feature_list, feature_importance):
	# 设定阈值
	fi_threshold = 5
	# 重要特征的下标
	important_idx = np.where(feature_importance > fi_threshold)[0]
	# 特征名称
	important_features = [feature_list[i] for i in important_idx]
	# 重要特征
	sorted_idx = np.argsort(feature_importance[important_idx])
	# 可视化
	pos = np.arange(sorted_idx.shape[0]) + 0.5
	plt.subplot(1,2,2)
	plt.title('Feature Importance')
	plt.barh(pos, feature_importance[important_idx][sorted_idx], height=0.5, color='r',align='center')
	plt.yticks(pos, [important_features[i] for i in sorted_idx])
	plt.xlabel('Relative Importance')
	plt.draw()
	plt.show()

# 读入训练集和测试集
dtest = pd.read_csv("test.csv")
dtrain = pd.read_csv("train.csv")
# 探索数据
# 查看数据集信息：查看后发现age字段和Embarked，Cabin字段有信息缺失,train的Fare有0，Test的Fare有0和na
print(dtrain.info())
print(dtest.info())
# 查看具体数值数据信息，未能发现很多有价值的信息，得到获救率大概38%，票价的方差较大，且75%的乘客票价低于平均票价
print(dtrain.describe())
# 查看离散数据类型分布,keyi kanchu 女性乘客偏多，大概占2/3，船舱分类比较多，很难作为因素进行分析，否则需要将分类汇总为大类，登陆最多的港为“S”
print(dtrain.describe(include=["O"]))
# 处理缺失值
# 使用平均年龄填充年龄中的nan值
dtrain["Age"].fillna(dtrain["Age"].mean(), inplace=True)
dtest["Age"].fillna(dtrain["Age"].mean(), inplace=True)
# 先把票价na的以0填充，再把平均值替换0值
dtrain["Fare"].fillna(0, inplace=True)
dtest["Fare"].fillna(0, inplace=True)
dtrain["Fare"].replace(0, dtrain["Fare"].mean(), inplace=True)
dtest["Fare"].replace(0, dtest["Fare"].mean(), inplace=True)
# 使用登陆最多的港口填充nan值
dtrain["Embarked"].fillna("S", inplace=True)
# 数据清洗完毕
# # 特征选择
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
trainsets = dtrain[features]
trainlb = dtrain["Survived"]
testsets = dtest[features]
# 将非数值型的特征转换为0/1的数值型
devc = DictVectorizer(sparse=False)
trainsets = devc.fit_transform(trainsets.to_dict(orient="record"))
# 生成新的训练集，并添加列名
newtrainsets = pd.DataFrame(trainsets, columns=devc.feature_names_)
# 使用LR回归，并训练，使用coef_可以进行相关性分析
clf = LogisticRegression(max_iter=200, verbose=True, random_state=33, tol=1e-4)
clf.fit(trainsets, trainlb)
# 获取特征重要性
features_importance = clf.coef_[0]
features_importance = 100 * features_importance / features_importance.max()
show_feature_importance(devc.feature_names_, features_importance)
# 根据重要性选取港口和性别作为特征进行分类预测
newtrainsets = newtrainsets[["Sex=female", "Sex=male", "Embarked=C", "Embarked=Q", "Embarked=S"]]
# 写个函数看看分类器对输入的要求
X_train, X_test, y_train, y_test = train_test_split(newtrainsets, trainlb, random_state=8)
clf1 = GaussianNB()
clf1.fit(X_train, y_train)
print("使用高斯贝叶斯模型预测的模型得分为：", clf1.score(X_test, y_test))
clf2 = tree.DecisionTreeClassifier(max_depth=4)
clf2.fit(X_train, y_train)
print("使用决策树分类器模型预测的模型得分为：", clf2.score(X_test, y_test))
clf3 = KNeighborsClassifier(n_neighbors=3)
clf3.fit(X_train, y_train)
print("使用K最近邻模型预测的模型得分为：", clf3.score(X_test, y_test))
newpd = pd.DataFrame(X_test, columns=["Sex=female", "Sex=male", "Embarked=C", "Embarked=Q", "Embarked=S"])
newpd["Survied"] = y_test
filterboolen = ((newpd["Sex=female"]==1) & (newpd["Survied"]==0))|((newpd["Sex=male"]==1) & (newpd["Survied"]==1))
print(newpd.shape, newpd[filterboolen].shape)
# 通过初步推测发现各个模型之间预测得分类似的原因很有可能是因为只是将性别作为了唯一判别因素。