import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm

# Simulate a representative dataset
np.random.seed(42)
n = 530

data = pd.DataFrame({
    'training_attended': np.random.binomial(1, 0.4, size=n),
    'additional_contributions': np.random.binomial(1, 0.2, size=n),
    'income_replacement_ratio': np.random.beta(2, 5, size=n) * 0.8
})

# Calculate the binary target variable probabilistically
# Using a logistic formula similar to your trained coefficients
log_odds = (
    -6.25
    + 0.33 * data['training_attended']
    - 0.34 * data['additional_contributions']
    + 15.55 * data['income_replacement_ratio']
)
prob_adequate = 1 / (1 + np.exp(-log_odds))
data['adequate_income'] = np.random.binomial(1, prob_adequate)

# Prepare data for modeling
X = data[['training_attended', 'additional_contributions', 'income_replacement_ratio']]
X = sm.add_constant(X)
y = data['adequate_income']

# Fit logistic regression model using statsmodels
model = sm.Logit(y, X).fit(disp=0)

# Convert to scikit-learn-style predict_proba by creating a wrapper
class SklearnLikeLogitModel:
    def __init__(self, sm_model):
        self.model = sm_model

    def predict_proba(self, X_input):
        if isinstance(X_input, pd.DataFrame):
            X_input = sm.add_constant(X_input, has_constant='add')
        elif isinstance(X_input, np.ndarray):
            X_input = sm.add_constant(X_input, has_constant='add')
        probs = self.model.predict(X_input)
        return np.vstack([1 - probs, probs]).T

# Wrap and save model
wrapped_model = SklearnLikeLogitModel(model)
joblib.dump(wrapped_model, 'logistic_model.pkl')
print("✅ Logistic model saved as 'logistic_model.pkl'")