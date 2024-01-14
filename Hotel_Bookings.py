import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
import xgboost
import streamlit as st


st.title('Hotel Bookings')
st.markdown(
    """
    Welcome to our app to predict the average daily rate of a hotel booking.
    ### Raw Data Source
    Check out this [Kaggle Page](https://www.kaggle.com/datasets/mojtaba142/hotel-booking) for the source of the raw data. 


"""
)
st.sidebar.markdown("# Hotel Bookings")
st.subheader('Raw data')

@st.cache_data
def load_data():
    data = pd.read_csv("C:\\Users\\Yrja\\Desktop\\Micro Degree AI\\Jaar4\\Cloud for AI\\Assignment\\Tasks\\hotel_booking.csv")
    return data

data_load_state = st.text('Loading data...')
rawdata = load_data()
data_load_state.text('Loading data...done!')
if st.checkbox("Show raw data first 5 rows"):
    st.write(rawdata.head())



if 'rawdata' not in st.session_state:
    st.session_state['rawdata'] = rawdata


##Descriptive
##data cleaning
descriptive_data = rawdata
descriptive_data['total_revenues'] = descriptive_data['adr'] * (descriptive_data['stays_in_weekend_nights'] + descriptive_data['stays_in_week_nights'])



descriptive_data['arrival_date_month'] = pd.to_datetime(descriptive_data.arrival_date_month, format='%B').dt.month
descriptive_data['reservation_status_date']=descriptive_data['reservation_status_date'].astype('datetime64[ns]')
descriptive_data.drop(['email', 'credit_card', 'phone-number', 'name','agent', 'company'], axis=1, inplace = True)
descriptive_data['children'].fillna(descriptive_data['children'].median(), inplace=True)
descriptive_data['children']=descriptive_data['children'].astype(int)
mode_country=descriptive_data['country'].mode()[0]
descriptive_data['country'] = descriptive_data['country'].fillna(mode_country)



if 'descriptive_data' not in st.session_state:
    st.session_state['descriptive_data'] = descriptive_data

@st.cache_data
def factorize_columns(data, columns_to_factorize):
    encoded_mappings = {}

    for column in columns_to_factorize:
        codes, unique_values = pd.factorize(data[column])
        data[column] = codes
        encoded_mappings[column] = {'codes': codes, 'unique_values': unique_values}

    return data, encoded_mappings

@st.cache_data
def replace_undefined_with_mode(data, column_name):
    mode_value = data[column_name].mode()[0]  # Get the mode (most frequent value)

    # Replace "undefined" values with the mode
    data[column_name] = data[column_name].replace('', mode_value)

    return data

data, mappings = factorize_columns(descriptive_data, ['hotel','meal','market_segment','distribution_channel','reserved_room_type','assigned_room_type', 'deposit_type', 'customer_type', 'reservation_status'])


modeldata = descriptive_data
modeldata.drop(['country','total_revenues','total_stay_in_nights'], axis=1, inplace = True)


if 'modeldata' not in st.session_state:
    st.session_state['modeldata'] = modeldata


#Model data