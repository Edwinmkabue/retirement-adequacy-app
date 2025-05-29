# model_wrapper.py
import numpy as np
import statsmodels.api as sm

class SklearnLikeLogitModel:
    def __init__(self, sm_model):
        self.model = sm_model

    def predict_proba(self, X_input):
        if isinstance(X_input, np.ndarray):
            X_input = sm.add_constant(X_input, has_constant='add')
        return np.vstack([1 - self.model.predict(X_input), self.model.predict(X_input)]).T
