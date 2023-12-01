from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import numpy as np

def R2Score():
	def metric(y_pred, y_true):
		return r2_score(y_true, y_pred)
	return metric

def MAE():
	def metric(y_pred, y_true):
		return mean_absolute_error(y_true, y_pred)
	return metric

def RMSE():
	def metric(y_pred, y_true):
		return np.sqrt(mean_squared_error(y_true, y_pred))
	return metric

def MAPE():
	def metric(y_pred, y_true):
		return mean_absolute_percentage_error(y_true, y_pred)
	return metric

def BIAS():
	def metric(y_pred, y_true):
		return np.mean(y_pred - y_true)
	return metric
