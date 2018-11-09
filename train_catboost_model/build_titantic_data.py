# -*- coding: utf-8 -*-

import re

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, CatBoostClassifier, Pool, cv

def trans_2_titile(name):
	item = re.compile('[,.]\s+').split(name)[1]
	if item in ['Ms', 'Mme', 'Mlle']:
		return 'Mlle'
	elif item in ['Capt', 'Don', 'Major', 'Sir']:
		return 'Sir'
	elif item in ['Dona', 'Lady', 'the Countess', 'Jonkheer']:
		return 'Lady'
	else:
		return item

def ticket_count(ticket):
	ticket_group = ticket.groupby(ticket).count()
	share_ticket_name = ticket_group[ticket_group > 1].index
	return ['share' if item in share_ticket_name else 'unique' for item in ticket]

def predict_age(age_data):
	X_train_data = age_data[age_data['Age'].notna()]
	y_train = X_train_data['Age']
	X_train = X_train_data.drop(columns=['PassengerId', 'Age'])

	categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title']
	columns = X_train.columns.tolist()
	categorical_features_indices = [columns.index(item) for item in categorical_features]
	_ = Pool(X_train, y_train, cat_features=categorical_features_indices)

	params = {'iterations': 50, 'eval_metric': 'RMSE', 'logging_level': 'Silent', 'l2_leaf_reg': 2.0, 'depth': 10, 'learning_rate': 0.007590561479761263}
	reg = CatBoostRegressor(**params)
	reg.fit(X_train, y_train, cat_features=categorical_features_indices)

	X_predict_data = age_data[age_data['Age'].isna()]
	X_predict = X_predict_data.drop(columns=['PassengerId', 'Age'])
	y_predict = reg.predict(X_predict)

	return X_predict_data.index, y_predict

def handle_data(data):
	data['Title'] = data['Name'].apply(trans_2_titile)
	data['age_section'] = data['Age'].fillna(-1).apply(lambda item: int(item / 10) + 1)
	data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
	data['TicketCount']  = ticket_count(data['Ticket'])
	data['Cabin'] = data['Cabin'].fillna('')
	data['Cabin_Area']  = data['Cabin'].apply(lambda item: item[0] if item != '' else 'X')
	data['Embarked'] = data['Embarked'].fillna('C')
	if len(data[data['Age'].isna()]) != 0:
		row_indexer, age_value = predict_age(data[['PassengerId', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'Age']])
		data.loc[row_indexer, 'Age'] = age_value
	
	data.drop(columns=['Name', 'Ticket', 'Cabin', 'age_section'], inplace=True)

	return data

def save_transposed_data():
	train_data = pd.read_csv("../titantic_data/train.csv")
	train_data = handle_data(train_data)
	train_data.to_csv("../titantic_data/transposed.csv", index=False)

if __name__ == '__main__':
	save_transposed_data()