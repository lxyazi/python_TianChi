import pandas as pd
import numpy as np
import sklearn
# import xgboost as xgb
# from sklearn.cross_validation import train_test_split


# 修改性别函数
def sex_change(sex_column):
	result = []
	for item in sex_column:
		if item == '男':
			result.append(0)
		else:
			result.append(1)
	return result


# 处理缺失值
def miss_value_to_avg(column):
	column = []
	result = np.array(column)
	i = 0
	sum = 0
	for value in result:
		if value == value:
			sum += value
			i += 1
	avg = sum / i
	result /= avg
	return result


origin_data = pd.read_csv('d_train_sex_change.csv', encoding='gb2312')
Y_column = origin_data.血糖
X_column = origin_data.drop(['id', '体检日期','血糖'], axis=1)
print(X_column)


# sex_column = origin_data['性别']
# temp = []
# origin_data['性别'] = temp
# origin_data.to_csv("d_train_sex_change.csv", encoding='gb2312', index=False)
