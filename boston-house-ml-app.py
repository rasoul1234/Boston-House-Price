import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import shap

st.write("""
# Boston House Price Prediction App

This app predicts the **Boston House Price**!
""")
st.write('---')

# Load the Boston House Price Dataset from GitHub
csv_url = 'https://raw.githubusercontent.com/rasoul1234/Boston-House-Price/main/BostonHouse.csv'
df = pd.read_csv(csv_url)
X = df.drop('Price', axis=1)
Y = df['Price']

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    CRIM = st.sidebar.slider('CRIM', float(X.CRIM.min()), float(X.CRIM.max()), float(X.CRIM.mean()))
    ZN = st.sidebar.slider('ZN', float(X.ZN.min()), float(X.ZN.max()), float(X.ZN.mean()))
    INDUS = st.sidebar.slider('INDUS', float(X.INDUS.min()), float(X.INDUS.max()), float(X.INDUS.mean()))
    
    # Use selectbox for CHAS (categorical variable)
    CHAS = st.sidebar.selectbox('CHAS (1 = near river; 0 = otherwise)', [0, 1], index=int(X.CHAS.mean()))
    
    NOX = st.sidebar.slider('NOX', float(X.NOX.min()), float(X.NOX.max()), float(X.NOX.mean()))
    RM = st.sidebar.slider('RM', float(X.RM.min()), float(X.RM.max()), float(X.RM.mean()))
    AGE = st.sidebar.slider('AGE', int(X.AGE.min()), int(X.AGE.max()), int(X.AGE.mean()), step=1)
    DIS = st.sidebar.slider('DIS', float(X.DIS.min()), float(X.DIS.max()), float(X.DIS.mean()))
    RAD = st.sidebar.slider('RAD', int(X.RAD.min()), int(X.RAD.max()), int(X.RAD.mean()), step=1)
    TAX = st.sidebar.slider('TAX', int(X.TAX.min()), int(X.TAX.max()), int(X.TAX.mean()), step=1)
    PTRATIO = st.sidebar.slider('PTRATIO', float(X.PTRATIO.min()), float(X.PTRATIO.max()), float(X.PTRATIO.mean()))
    B = st.sidebar.slider('B', float(X.B.min()), float(X.B.max()), float(X.B.mean()))
    LSTAT = st.sidebar.slider('LSTAT', float(X.LSTAT.min()), float(X.LSTAT.max()), float(X.LSTAT.mean()))
    
    data = {
        'CRIM': CRIM,
        'ZN': ZN,
        'INDUS': INDUS,
        'CHAS': CHAS,
        'NOX': NOX,
        'RM': RM,
        'AGE': AGE,
        'DIS': DIS,
        'RAD': RAD,
        'TAX': TAX,
        'PTRATIO': PTRATIO,
        'B': B,
        'LSTAT': LSTAT
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
df_input = user_input_features()

# Main Panel

# Display specified input parameters
st.header('Specified Input parameters')
st.write(df_input)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)

# Apply Model to Make Prediction
prediction = model.predict(df_input)

st.header('Prediction of MEDV (Median Value of Owner-Occupied Homes in $1000)')
st.write(prediction)
st.write('---')

# Feature Importance with SHAP
st.header('Feature Importance')

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, X)
st.pyplot(plt.gcf())
st.write('---')

# SHAP bar plot
plt.figure()
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(plt.gcf())
