from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, median_absolute_error,mean_tweedie_deviance,d2_tweedie_score

import numpy as np

def R2Score():
	def metric(y_pred, y_true):		
		return r2_score(y_true, y_pred)
	return metric

def MAE():
	def metric(y_pred, y_true):
		return mean_absolute_error(y_true, y_pred)
	return metric

def MedAE():
	def metric(y_pred, y_true):
		return median_absolute_error(y_true, y_pred)
	return metric

def RMSE():
	def metric(y_pred, y_true):
		return np.sqrt(mean_squared_error(y_true, y_pred))
	return metric

def rRMSE():
	def metric(y_pred, y_true):
		return np.sqrt(mean_squared_error(y_true, y_pred))/np.mean(y_true)
	return metric

def MAPE():
	def metric(y_pred, y_true):
		return mean_absolute_percentage_error(y_true, y_pred)
	return metric

def BIAS():
	def metric(y_pred, y_true):
		return np.mean(y_pred - y_true)
	return metric

def TweediePoisson():
	def metric(y_pred, y_true):
		return mean_tweedie_deviance(y_true, y_pred, power=1)
	return metric

def D2score():
	def metric(y_pred, y_true):
		return d2_tweedie_score(y_true, y_pred, power=1)
	return metric

def PCorr():
	def metric(y_pred, y_true):
		y_pred_mean = y_pred.mean()
		y_pred_std = y_pred.std()
		y_true_mean = y_true.mean()
		y_true_std = y_true.std()

		covariance_ = np.mean((y_pred-y_pred_mean)*(y_true-y_true_mean))

		return covariance_/(y_pred_std*y_true_std)
	return metric