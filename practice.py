import pandas as pd
import numpy as np

#action1:求2+4+6+8+...+100的求和，用Python该如何写
print("结果为：", sum(range(2, 101, 2)))

# action2 统计全班的成绩
# 班里有5名同学，现在需要你用Python来统计下这些人在语文、英语、数学中的平均成绩、最小成绩、最大成绩、方差、标准差。
scores = pd.read_excel("scores.xlsx", encoding="utf-8")
print(scores.describe().loc[["mean", "std", "min", "max"]])
# 然后把这些人的总成绩排序，得出名次进行成绩输出（可以用numpy或pandas）
scores["总成绩"] = scores[["语文", "数学", "英语"]].sum(axis=1)
scores = scores.sort_values("总成绩", ascending=False)
scores["排名"] = np.arange(1, scores.shape[0]+1)
print(scores)

#Action3: 对汽车质量数据进行统计
#数据集：car_complain.csv
#600条汽车质量投诉
#Step1，数据加载
dfc = pd.read_csv("car_complain.csv", encoding="utf-8")
#Step2，数据预处理
dfc["brand"] = dfc["brand"].apply(lambda x:x.replace("-", ""))
#拆分problem类型 => 多个字段,这里不太懂分成多字段的意思和操作，统计单行抱怨中所含的问题点个数
dfc["prob_count"] = dfc["problem"].apply(lambda x: len(x.split(",")) - 1)
#Step3，数据统计
#对数据进行探索：品牌投诉总数，车型投诉总数
print(dfc.groupby("brand")["prob_count"].sum().sort_values(ascending=False))
print(dfc.groupby("car_model")["prob_count"].sum().sort_values(ascending=False))
#哪个品牌的平均车型投诉最多
print((dfc.groupby("brand")["prob_count"].sum()/dfc.groupby("brand")["car_model"].agg(set).map(len)).sort_values(ascending=False))







