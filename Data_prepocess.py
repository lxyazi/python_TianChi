import numpy as np
import pandas as pd


def sex_change(data):
	temp = []
	sex_column = data['性别']
	for value in sex_column:
		if value == '男':
			temp.append(1)
		else:
			temp.append(0)
	data['性别'] = temp


def column_miss_value_change(column):
	sum = 0
	count = 0
	for value in column:
		if value == value:
			sum = sum + value
			count += 1

	avg = sum / count

	temp = []
	for value in column:
		if value != value:
			value = avg
		temp.append(value)
	return temp


origin_data = pd.read_csv("d_train_20180102.csv", encoding="gb2312")
sex_change(origin_data)
for value in ['*天门冬氨酸氨基转换酶','*丙氨酸氨基转换酶',
              '*碱性磷酸酶','*r-谷氨酰基转换酶','*总蛋白',
              '白蛋白','*球蛋白','白球比例','甘油三酯','总胆固醇',
              '高密度脂蛋白胆固醇','低密度脂蛋白胆固醇','尿素',
              '肌酐','尿酸','乙肝表面抗原','乙肝表面抗体','乙肝e抗原',
              '乙肝e抗体','乙肝核心抗体','白细胞计数','红细胞计数',
              '血红蛋白','红细胞压积','红细胞平均体积','红细胞平均血红蛋白量',
              '红细胞平均血红蛋白浓度','红细胞体积分布宽度','血小板计数',
              '血小板平均体积','血小板体积分布宽度','血小板比积','中性粒细胞%',
              '淋巴细胞%','单核细胞%','嗜酸细胞%','嗜碱细胞%']:
	column = origin_data[value]
	temp = column_miss_value_change(column)
	origin_data[value] = temp

	origin_data.to_csv('d_train_change.csv', encoding='gb2312')
