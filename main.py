import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
# %matplotlib inline
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.tree import _tree
from sklearn.metrics import accuracy_score
from chefboost import Chefboost as chef
from sklearn.datasets import load_iris


with open('bank.csv') as f:
	df = pd.read_csv(f, dtype={'age': np.int64, 'job': np.object_,'education': np.object_,'default': np.object_, 'balance': np.int64, 'loan': np.object_, 'contact': np.object_, 'day': np.int64, 'month': np.object_,'duration':np.float64,'campaign': np.int64 , 'pdays': np.float64, 'previous':np.int64,'poutcome': np.object_, 'y': np.object_})
print(df.head(12))
df.info()
print(df.describe()) #thống kê các thuộc tính đính lượng như: đếm số giá trị, gtln, gtnn,..

print(list(df.columns))
# biểu đồ
sns.barplot(x="y", y="age", data=df)
#plt.show()

#rời rạc thuộc tính age
minage = df['age'].min()
maxage = df['age'].max()
interval = (maxage-minage)/4
bin1 = minage + interval
bin2 = bin1 + interval
bin3 = bin2 + interval

for dataset in [df]:
	dataset['age'] = dataset['age'].astype(int)
	dataset.loc[dataset['age'] <= bin1, 'age'] = 0
	dataset.loc[(dataset['age'] > bin1) & (dataset['age']<= bin2), 'age'] = 1
	dataset.loc[(dataset['age'] > bin2)& (dataset['age'] <= bin3), 'age'] = 2
	dataset.loc[(dataset['age'] > bin3), 'age'] = 3

print(df['age'].value_counts())

#rời rạc thuộc tính balance
pd.qcut(df['balance'],4).value_counts()

for dataset in [df]:
	dataset['balance'] = dataset['balance'].astype(int)
	dataset.loc[dataset['balance'] <= 69.0, 'balance'] = 0
	dataset.loc[(dataset['balance'] > 69.0) & (dataset['balance']<= 444.0), 'balance'] = 1
	dataset.loc[(dataset['balance'] > 444.0) & (dataset['balance']<= 1480.0), 'balance'] = 2
	dataset.loc[(dataset['balance'] > 1480.0) , 'balance'] = 3

print(df['balance'].value_counts())


#rời rạc thuộc tính campaign
mincampaign = df['campaign'].min()
maxcampaign = df['campaign'].max()
interval = (maxcampaign-mincampaign)/4
bin1 = mincampaign + interval
bin2 = bin1 + interval
bin3 = bin2 + interval
for dataset in [df]:
	dataset['campaign'] = dataset['campaign'].astype(int)
	dataset.loc[dataset['campaign'] <= bin1, 'campaign'] = 0
	dataset.loc[(dataset['campaign'] > bin1) & (dataset['campaign']<= bin2), 'campaign'] = 1
	dataset.loc[(dataset['campaign'] > bin2) & (dataset['campaign']<= bin3), 'campaign'] = 2
	dataset.loc[(dataset['campaign'] > bin3), 'campaign'] = 3
print(df['campaign'].value_counts())

#rời rạc thuộc tính previous
minprevious = df['previous'].min()
maxprevious = df['previous'].max()
interval = (maxprevious-minprevious)/3
bin1 = minprevious + interval
bin2 = bin1 + interval
#print(bin1, bin2, minage, interval)

for dataset in [df]:
	dataset['previous'] = dataset['previous'].astype(int)
	dataset.loc[dataset['previous'] <= bin1, 'previous'] = 0
	dataset.loc[(dataset['previous'] > bin1) & (dataset['previous']<= bin2), 'previous'] = 1
	dataset.loc[(dataset['previous'] > bin2), 'previous'] = 2
print(df['previous'].value_counts())

#rời rạc thuộc tính pdays

maxpdays = df['pdays'].max()
interval = (maxpdays)/3
bin1 = interval
bin2 = bin1 + interval

for dataset in [df]:
	dataset['pdays'] = dataset['pdays'].astype(int)
	dataset.loc[dataset['pdays'] == -1, 'pdays'] = 0
	dataset.loc[(dataset['pdays'] > 0) & (dataset['pdays']<= bin1), 'pdays'] = 1
	dataset.loc[(dataset['pdays'] > bin1) & (dataset['pdays']<= bin2), 'pdays'] = 2
	dataset.loc[(dataset['pdays'] > bin2), 'pdays'] = 3

print(df['pdays'].value_counts())

#rời rạc duration
print(pd.qcut(df['duration'],4).value_counts())

for dataset in [df]:
	dataset['duration'] = dataset['duration'].astype(int)
	dataset.loc[dataset['duration'] <= 104.0, 'duration'] = 0
	dataset.loc[(dataset['duration'] > 104.0) & (dataset['duration']<= 185.0), 'duration'] = 1
	dataset.loc[(dataset['duration'] > 185.0) & (dataset['duration']<= 329.0), 'duration'] = 2
	dataset.loc[(dataset['duration'] > 329.0), 'duration'] = 3

print(df['duration'].value_counts())

#rời rạc thuộc tính day
minday = df['day'].min()
maxday = df['day'].max()
interval = (maxday-minday)/3
bin1 = minday + interval
bin2 = bin1 + interval
#print(bin1, bin2, minage, interval)

for dataset in [df]:
	dataset['day'] = dataset['day'].astype(int)
	dataset.loc[dataset['day'] <= bin1, 'day'] = 0
	dataset.loc[(dataset['day'] > bin1) & (dataset['day']<= bin2), 'day'] = 1
	dataset.loc[(dataset['day'] > bin2), 'day'] = 2
print(df['day'].value_counts())

#xoa thuoc tinh



#khao sat do tuong dong
fig = plt.figure(figsize=(16,9))
sns.heatmap(df.corr(method='pearson'),annot=True)
#fig.show()
# tach cot du lieuj
features = df.drop('y', axis=1)
labels = df['y']

# grid = sns.FacetGrid(df, col='y', row='age', height=4.2, aspect=1.6)
# grid.map(plt.hist, 'age', alpha=.5, bins = 10)
# grid.add_legend()

#chuyen ve dang one-hot
features.select_dtypes(exclude=['int64']).columns
features_onehot = pd.get_dummies(features, columns=features.select_dtypes(exclude=['int64']).columns)
print(features_onehot)

#tach thanhf test va train

x_train, x_test, y_train, y_test = train_test_split(features_onehot,labels,test_size = 0.50, random_state = 42)

clfID3 = tree.DecisionTreeClassifier(criterion="entropy", random_state=0)
#train decision tree
clfID3.fit(x_train, y_train)
text_representation = tree.export_text(clfID3)
print(text_representation)
#duự đoán test
tree_predID3 = clfID3.predict(x_test)

#===============================
clfC45 = tree.DecisionTreeClassifier(criterion="gini", random_state=0)
# tạo model
#max_depth = 3 là độ sâu của cây quyết định
clfC45 = clfC45.fit(x_train, y_train)

tree_predC45 = clfC45.predict(x_test)
# Kết quả suy đoán
tree_scoreC45 = metrics.accuracy_score(y_test, tree_predC45)

print("Accuracy C4.5: ",tree_scoreC45)
print("Report: ", metrics.classification_report(y_test, tree_predC45))
print(type(metrics.classification_report(y_test, tree_predC45)))
#ma tran nham lan
tree_cmC45 = metrics.confusion_matrix(y_test, tree_predC45)
print(tree_cmC45)


#===========================================

#độ chính xác
tree_scoreID3 = metrics.accuracy_score(y_test, tree_predID3)

print("Accuracy ID3: ", tree_scoreID3)
print("Report: ", metrics.classification_report(y_test, tree_predID3))
print(type(metrics.classification_report(y_test, tree_predID3)))
#ma tran nham lan
tree_cmID3 = metrics.confusion_matrix(y_test, tree_predID3)
print(tree_cmID3)


#bieu dien ma tran nham lan
plt.figure(figsize=(12,12))
sns.heatmap(tree_cmID3, annot=True, fmt=".3f", linewidths = .5, square=True, cmap='Blues_r')
plt.xlabel('Lớp dự đoán từ mô hình')
plt.ylabel('Lớp trên thực tế')
title = "Độ chính xác của cây quyết định: {0}".format(tree_scoreID3)
plt.title(title, size=15)
plt.show()
#
# def get_code(tree, feature_names):
#         left = tree.tree_.children_left
#         right = tree.tree_.children_right
#         threshold = tree.tree_.threshold
#         features  = [feature_names[i] for i in tree.tree_.feature]
#         value = tree.tree_.value
#
#         def recurse(left, right, threshold, features, node):
#                 if (threshold[node] != -2):
#                         print ("if ( " + features[node] + " <= " + str(threshold[node]) + " ) {")
#                         if left[node] != -1:
#                                 recurse (left, right, threshold, features,left[node])
#                         print ("} else {")
#                         if right[node] != -1:
#                                 recurse (left, right, threshold, features,right[node])
#                         print ("}")
#                 else:
#                         print ("return ") + str(value[node])
#
#         recurse(left, right, threshold, features, 0)
#
# get_code(clfC45, df.columns )
 #ve cay ID3
fig, ax = plt.subplots(figsize=(50,24))
tree.plot_tree(clf, filled=True, fontsize=10)
plt.savefig('decision_tree', dpi=100)
plt.show()