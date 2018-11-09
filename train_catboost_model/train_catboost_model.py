# -*- coding: utf-8 -*-

from sklearn.externals import joblib
from catboost import CatBoostRegressor, CatBoostClassifier, Pool, cv
import pandas as pd


def build_train_data():
	data = pd.read_csv("../titantic_data/transposed.csv")
	y_train = data['Survived']
	X_train = data.drop(columns=['PassengerId', 'Survived'], axis=1)

	categorical_columns = ['Pclass', 'Sex', 'Embarked', 'Title', 'TicketCount', 'Cabin_Area']

	columns = X_train.columns.tolist()
	categorical_features_indices = [columns.index(item) for item in categorical_columns]
	# print categorical_features_indices

	return X_train, y_train, categorical_features_indices


def train_model():
	X_train, y_train, categorical_features_indices = build_train_data()
	params = {'iterations': 50, 'eval_metric': 'AUC', 'logging_level': 'Silent', 'l2_leaf_reg': 1.0, 'depth': 4, 'learning_rate': 0.07877005965234678}
	clf = CatBoostClassifier(**params)
	clf.fit(X_train, y_train, cat_features=categorical_features_indices)

	joblib.dump(clf, './catboost_titantic.model')

	return clf


def dump_python_file(model, filename='catboost_titantic'):
	X_train, y_train, categorical_features_indices = build_train_data()
	train_pool = Pool(X_train, label=y_train, cat_features=categorical_features_indices)
	model.save_model(filename + '.py', format="python", pool=train_pool)


if __name__ == '__main__':
	model = train_model()
	dump_python_file(model)
