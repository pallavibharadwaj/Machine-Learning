import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.figure(figsize=(20,20))

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
from sklearn import metrics
from sklearn.model_selection import train_test_split

DATA_PATH="churn_dataset.csv"

def missing_values(df):
	missing_cols = []
	missing_count = 0
	for col in df.columns.tolist():
		# replace empty strings with Null
		df[col] = df[col].replace(r' ', np.nan)

		missing = df[col].isnull().sum()
		if(missing):
			missing_cols.append(col)
			missing_count += missing

	# columns having missing values
	# TotalCharges is the only column with missing values
	print("Columns with missing values: %s" %(missing_cols))

	# percentage of missing values
	total = df.shape[0] * df.shape[1]
	percent = (missing_count/total) * 100
	print("Percentage of missing values: %.4f" %(percent))

	# replace missing values with mean of the column
	# TotalCharges is a numeric feature
	for col in missing_cols:
		# convert to numeric feature
		df[col] = pd.to_numeric(df[col], downcast="float")

		mean = df[col].mean()
		df[col] = df[col].replace(np.nan, mean)

	return df


class feature_selection:

	def __init__(self, df):
		# encode categorical features
		le = preprocessing.LabelEncoder()
		self.df = df.apply(le.fit_transform)


	def corr_matrix(self):
		corrmat = self.df.corr()

		# plot heat map of correlation matrix
		sns.heatmap(corrmat, annot=True)
		plt.show()

		# drop columns with nearly zero correlation with target
		unrelated = corrmat[abs(corrmat) < 0.05]['Churn']
		unrelated_cols = unrelated[~np.isnan(unrelated)].keys()
		df = self.df.drop(columns=unrelated_cols)

		return df


	def backward_elimination(self, df, p_value):
		# features and labels
		# all features are selected initially
		features = df.iloc[:, 0:-1].values
		labels = df.iloc[:, -1].values
		columns = df.columns[:-1]

		# iteratively remove each column that does not influence the target
		for i in range(0, len(columns)):
			# fit a regressor model - Ordinary Least Squares
			regressor = sm.OLS(labels, features).fit()

			# find feature with maximim p_value
			max_p = max(regressor.pvalues)

			# remove feature if p_value is greater than threshold
			if max_p > p_value:
				index = np.where(regressor.pvalues==max_p)
				features = np.delete(features, index, axis=1)
				columns = np.delete(columns, index)

		return (features, columns, labels)


def classifier(X, Y):
	# 90:10 train-test ratio
	x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

	model = RandomForestClassifier(n_estimators=400, max_depth=4)
	model.fit(x_train, y_train)

	y_pred = model.predict(x_test)

	accuracy = metrics.accuracy_score(y_test, y_pred, normalize=True)
	print("Test Accuracy = %.4f" % (accuracy))


def main():
	# load the data
	df = pd.read_csv(DATA_PATH)
	print(df)

	# Task 1: handle missing values
	df = missing_values(df)

	# Task 2: feature selection
	fs = feature_selection(df)
	# correlation matrix
	df = fs.corr_matrix()
	# backward feature selection
	features, selected_columns, labels = fs.backward_elimination(df, p_value=0.05)

	print("Selected Columns: ", selected_columns)

	# Task 3: classifier model
	classifier(features, labels)

if __name__=='__main__':
	main()