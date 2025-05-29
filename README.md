# 💼 Retirement Adequacy Checker

This is a simple, research-backed web tool to estimate whether your current savings behavior is likely to lead to adequate retirement income — built using a logistic regression model trained on pension adequacy insights.

## 🔍 What It Does

- Simulates your future pension and income replacement ratio
- Uses a predictive model to estimate adequacy probability
- Provides tailored recommendations to improve your retirement outlook

## 📊 Inputs

- Current monthly salary
- Age and expected retirement age
- Savings behavior (contribution rate, training participation, voluntary contributions)

## 🚀 Try the App

🌐 [Click here to launch the app](https://retirement-adequacy.streamlit.app/)  

## 🧰 Tech Stack

- Python 3.12
- Streamlit
- statsmodels (logistic regression)
- joblib (model persistence)
- NumPy & Pandas

## 📦 Setup Locally

```bash
git clone https://github.com/yourusername/retirement-adequacy-app.git
cd retirement-adequacy-app
pip install -r requirements.txt
streamlit run retirement_adequacy_app.py
