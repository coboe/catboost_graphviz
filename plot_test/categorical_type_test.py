# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '../plot')

import os, imp
import pandas as pd
from sklearn.externals import joblib

from CatBoostCategoricalFeature import CatBoostCategoricalFeature
from CatBoostCategoricalModel import CatBoostCategoricalModel

def build_train_data():
	train_data = pd.read_csv("../titantic_data/transposed.csv")
	y_train = train_data['Survived']
	X_train = train_data.drop(columns=['PassengerId', 'Survived'])
	
	categorical_columns = ['Pclass', 'Sex', 'Embarked', 'Title', 'TicketCount', 'Cabin_Area']
	train_feature_columns =  X_train.columns.tolist()
	float_columns = [column for column in train_feature_columns if column not in categorical_columns]

	return float_columns, categorical_columns, X_train, y_train

def load_model(model_file):
	model = joblib.load(model_file)
	
	model_python_file = model_file.replace('.model', '.py')
	if not os.path.exists(model_python_file):
		model.save_model(model_python_file, format='python') # the api has change on catboost 0.10.*

	fp, pathname, description = imp.find_module(model_python_file.replace('.py', ''))
	catBoostPythonFile = imp.load_module('catBoostPythonFile', fp, pathname, description)

	return model, catBoostPythonFile


def value_export_file(model_file):

	model, catBoostPythonFile = load_model(model_file)
	float_columns, categorical_columns, X_train, _ = build_train_data()

	for index in range(X_train.shape[0]):
		data = X_train.loc[index]

		float_feature = data[float_columns].values
		cat_feature = data[categorical_columns].values
		# print float_feature, cat_feature

		predict_value = catBoostPythonFile.apply_catboost_model(float_feature, cat_feature)

		data = data.to_frame().T
		model_pridect_raw = model.predict(data, prediction_type='RawFormulaVal')
		
		if predict_value != model_pridect_raw[0]:
			print predict_value, model_pridect_raw[0]
	print 'End.'

def compare_value_test(model_file):

	_, catBoostPythonFile = load_model(model_file)

	import feature_info

	catgrory_feature = CatBoostCategoricalFeature(catBoostPythonFile, feature_info)
	catboost_tree = CatBoostCategoricalModel(catBoostPythonFile.catboost_model)

	float_columns, categorical_columns, X_train, _ = build_train_data()
	for index in range(X_train.shape[0]):
		data = X_train.loc[index]

		float_feature = data[float_columns].values
		cat_feature = data[categorical_columns].values

		value_1 = catBoostPythonFile.apply_catboost_model(float_feature, cat_feature)
		
		catgrory_feature.set_feature_value(float_feature, cat_feature)
		catboost_tree.set_binary_features(catgrory_feature.binary_features)

		value_2 = catboost_tree.get_result()
		if value_1 != value_2:
			print value_1, value_2
	print 'End.'

def draw_test(model_file):

	from graphviz import Digraph

	_, catBoostPythonFile = load_model(model_file)

	import feature_info

	catgrory_feature = CatBoostCategoricalFeature(catBoostPythonFile, feature_info)
	catboost_tree = CatBoostCategoricalModel(catBoostPythonFile.catboost_model)

	float_feature = [22.0, 1, 0, 7.25, 2]
	cat_feature = [3, 'male', 'S', 'Mr', 'unique', 'X']
	catgrory_feature.set_feature_value(float_feature, cat_feature)
	catboost_tree.set_binary_features(catgrory_feature.binary_features)

	for tree_id in range(catBoostPythonFile.catboost_model.tree_count):
		dot = Digraph(name='tree_' + str(tree_id))
		catgrory_feature.draw_feature(dot, catboost_tree.tree_info[tree_id]['feature_list'])
		catboost_tree.draw_sub_tree(dot, tree_id)
		dot.format = 'pdf'
		path = dot.render('./catboost_tree/' + str(tree_id), cleanup=True)
		print path

if __name__ == '__main__':
	model_file = 'catboost_titantic.model'
	value_export_file(model_file)
	compare_value_test(model_file)
	draw_test(model_file)