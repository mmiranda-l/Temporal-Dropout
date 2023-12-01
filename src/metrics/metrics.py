import numpy as np

from .metric_predictions import R2Score, MAPE, RMSE, MAE, BIAS

class BaseMetrics(object):
	"""central metrics class to provide standard metric types"""

	def __init__(self, task_type="classification", metric_types=[]):
		"""build BaseMetrics

		Parameters
		----------
		metric_types : list of str, optional
		    declaration of metric types to be used, by default all are used
		"""
		self.task_type = task_type
		self.metric_types = [v.lower() for v in metric_types]
		self.metric_dict = {}

	def __call__(self, prediction, target):
		"""call forward for each metric in collection

		Parameters
		----------
		prediction : array_like (n_samples, n_outputs)
		    prediction tensor
		target : array_like     (n_samples, n_outputs)
		    ground truth tensor, in classification: n_outputs=1, currently working only =1
		"""
		if not isinstance(prediction, np.ndarray):
			prediction = np.asarray(prediction)
		if not isinstance(target, np.ndarray):
			target = np.asarray(target)

		if self.task_type=="classification" and hasattr(self,"aux_metric"):
			self.n_samples = self.aux_metric(prediction,target)
		else:
			self.n_samples = []

		#forward over all metrics
		return {name: func(prediction, target) for (name, func) in self.metric_dict.items()}

	def get_metric_types(self):
		"""return list of metric types inside collection

		Returns
		-------
		list of strings
		"""
		return list(self.metric_dict.keys())

	def reverse_forward(self, target,prediction):
		return self(prediction, target)


class RegressionMetrics(BaseMetrics):
    def __init__(self, metric_types=["R2","RMSE","MAE", "MAPE", "BIAS"]):
        """build RegressionMetrics

        Parameters
        ----------
        metric_types : list of str, optional
            declaration of metric types to be used, by default all are used
        """
        super(RegressionMetrics,self).__init__("regression", metric_types)

        for metric in self.metric_types:
            if "r2"==metric:
                self.metric_dict["R2"] = R2Score()
            elif "mae"==metric:
                self.metric_dict["MAE"] = MAE()
            elif "rmse"==metric:
                self.metric_dict["RMSE"] = RMSE()
            elif "mape"==metric:
                self.metric_dict["MAPE"] = MAPE()
            elif "bias"==metric:
                self.metric_dict["BIAS"] = BIAS()

