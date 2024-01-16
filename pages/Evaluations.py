import streamlit as st
import joblib

st.markdown("# Evaluations...live edit ?")
st.sidebar.markdown("# Evaluations")

lr_model = joblib.load('models/lr_model.joblib')
xgb_model = joblib.load('models/xgb_model.joblib')
knn_model = joblib.load('models/knn_model.joblib')

