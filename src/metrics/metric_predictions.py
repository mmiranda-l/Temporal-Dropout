from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import scipy.stats as stats

import numpy as np

def calculate_bounds(prediction, logvar, confidence_level = 0.95):
	# Calculate the z-score for the specified confidence level
	z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
	lower_bound = (prediction - z * logvar)
	upper_bound = (prediction + z * logvar)
	return lower_bound, upper_bound

def picp(y_pred, y_true, logvar, confidence_level = 0.95, n_return_within=False):
	sigma = np.sqrt(np.exp(logvar))
	pu = np.std(y_pred, axis=1) + sigma.mean(axis=1)
	lower_bound, upper_bound = calculate_bounds(y_pred.mean(axis=1), pu, confidence_level=confidence_level)
	satisfies_upper = y_true <= upper_bound
	satisfies_lower = y_true >= lower_bound
	n_within = sum(satisfies_upper * satisfies_lower)
	if not n_return_within:
		return np.mean(satisfies_upper * satisfies_lower)
	else: 
		return np.mean(satisfies_upper * satisfies_lower), n_within

def R2Score():
	def metric(y_pred, y_true, **kwargs):
		if y_pred.ndim == 2: y_pred = y_pred.mean(axis=-1)
		return r2_score(y_true, y_pred)
	return metric

def MAE():
	def metric(y_pred, y_true, **kwargs):
		if y_pred.ndim == 2: y_pred = y_pred.mean(axis=-1)
		return mean_absolute_error(y_true, y_pred)
	return metric

def RMSE():
	def metric(y_pred, y_true, **kwargs):
		if y_pred.ndim == 2: y_pred = y_pred.mean(axis=-1)
		return np.sqrt(mean_squared_error(y_true, y_pred))
	return metric

def MAPE():
	def metric(y_pred, y_true, **kwargs):
		if y_pred.ndim == 2: y_pred = y_pred.mean(axis=-1)
		return mean_absolute_percentage_error(y_true, y_pred)
	return metric

def BIAS():
	def metric(y_pred, y_true, **kwargs):
		if y_pred.ndim == 2: y_pred = y_pred.mean(axis=-1)
		return np.mean(y_pred - y_true)
	return metric

def PICP():	
	def metric(y_pred, y_true, logvar, confidence_level = 0.95):
		return picp(y_pred, y_true, logvar, confidence_level)
	return 	metric

def ECE():
	def metric(y_pred, y_true, logvar):
		quantile_levels = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999])
		coverage = []
		n_within = []
		for alpha in quantile_levels:
			c, n = picp(y_pred, y_true, logvar, alpha, n_return_within=True)
			coverage.append(c)
			n_within.append(n)
		coverage, n_within = np.array(coverage), np.array(n_within)
		return 1/len(y_true) * (np.sum(n_within * (abs(coverage - quantile_levels))))

	return metric

def PU():
	def metric(y_pred, y_true, logvar):
		sigma2 = np.sqrt(np.exp(logvar))
		epistemic = np.std(y_pred, axis=-1)
		aleatoric = np.mean(sigma2, axis=-1)
		return epistemic.mean() + aleatoric.mean()
	return metric