import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from utils import load_csv_data, factorize_columns, RMSE, calculate_relative_error


st.markdown("# Evaluations")
st.sidebar.markdown("# Evaluations")

lr_model = st.session_state.get('lr_model', None)
xgb_model = st.session_state.get('xgb_model', None)
knn_model = st.session_state.get('knn_model', None)
data = st.session_state.get('df', None)

if lr_model is None:
    lr_model = joblib.load('models/lr_model.joblib')
    st.session_state['lr_model'] = lr_model

if xgb_model is None:
    xgb_model = joblib.load('models/xgb_model.joblib')
    st.session_state['xgb_model'] = xgb_model

if knn_model is None:
    knn_model = joblib.load('models/knn_model.joblib')
    st.session_state['knn_model'] = knn_model

if data is None:
    data = load_csv_data('hotel_booking.csv')
    st.session_state['df'] = data

#dit mag eveneens in zijn apparte functie in utils
y = data[['adr']]
X = data.drop(['adr', 'reservation_status_date','is_repeated_guest'],axis=1)

Xtrain, Xrest, ytrain, yrest = train_test_split(X, y, test_size=0.2)
Xval, Xtest, yval, ytest = train_test_split(Xrest, yrest, test_size=0.5)

##avoid dataleakage. filling in missing values after train test split
Xtrain['children'].fillna(Xtrain['children'].median(), inplace=True)
Xtest['children'].fillna(Xtest['children'].median(), inplace=True)
Xtrain['children']=Xtrain['children'].astype(int)
Xtest['children']=Xtest['children'].astype(int)

data, mappings = factorize_columns(data, ['hotel','meal','market_segment','distribution_channel','reserved_room_type','assigned_room_type', 'deposit_type', 'customer_type', 'reservation_status'])

default_params = {
    'hotel': 1,
    'is_canceled': 0,
    'lead_time': 0,
    'arrival_date_year': 2022,
    'arrival_date_month': 1,
    'arrival_date_week_number': 1,
    'arrival_date_day_of_month': 1,
    'stays_in_weekend_nights': 0,
    'stays_in_week_nights': 1,
    'adults': 2,
    'children': 2,
    'babies': 0,
    'meal': 0,
    'market_segment': 1,
    'distribution_channel': 1,
    'previous_cancellations': 0,
    'previous_bookings_not_canceled': 0,
    'reserved_room_type': 0,
    'assigned_room_type': 1,
    'booking_changes': 0,
    'deposit_type': 1,
    'days_in_waiting_list': 0,
    'customer_type': 0,
    'required_car_parking_spaces': 0,
    'total_of_special_requests': 0,
    'reservation_status': 1,
}

input_data = pd.DataFrame([default_params])

input = np.array(input_data).reshape(1, -1)

y_pred_lr = lr_model.predict(input)
y_pred_xgb = xgb_model.predict(input)
y_pred_knn = knn_model.predict(input)

def compare(lr_model, xgb_model, knn_model):

    st.title("Model Comparison")

    st.sidebar.header("Input Parameters")
    text_input = st.sidebar.text_input("Enter Text", "Default Text")

    st.header("Model Results")

    st.subheader("Root Mean Squared Error")
    st.write(f"Linear Regression: {rmse_lr}")
    st.write(f"XGBoost: {rmse_xgb}")
    st.write(f"KNN: {rmse_knn}")
  
    st.subheader("Relative Error")
    st.write(f"Linear Regression: {relative_error_lr}%")
    st.write(f"XGBoost: {relative_error_xgb}%")
    st.write(f"KNN: {relative_error_knn}%")

pred_comparison = pd.DataFrame({
    "Actual": ytest.median(),
    "Linear Regression": y_pred_lr[0],
    "XGBoost": y_pred_xgb[0],
    "KNN": y_pred_knn[0]
})
st.write(pred_comparison.head(10))