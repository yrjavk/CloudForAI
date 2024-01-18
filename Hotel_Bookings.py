import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st
import joblib

#load data
@st.cache_data
def load_csv_data(filename):
    data = pd.read_csv(filename)
    data.drop(['email', 'credit_card', 'phone-number', 'name'], axis=1, inplace = True)
    return data

#data cleaning for descriptive analysis 
@st.cache_data
def data_preparation(data):
    #drop columns. Too many missing values
    data.drop(['agent', 'company'], axis=1, inplace = True)
    #set correct datatype
    data['arrival_date_month'] = pd.to_datetime(data.arrival_date_month, format='%B').dt.month
    data['reservation_status_date']=data['reservation_status_date'].astype('datetime64[ns]')
    #add column total_revenues
    data['total_revenues'] = data['adr'] * (data['stays_in_weekend_nights'] + data['stays_in_week_nights'])
    data['total_stay_in_nights'] = data['stays_in_week_nights'] + data['stays_in_weekend_nights']
    # remove outlier
    data.drop(data['adr'].idxmax(), inplace = True)
    return data

#set categorical columns to numberical values
@st.cache_data
def factorize_columns(data, columns_to_factorize):
    encoded_mappings = {}

    for column in columns_to_factorize:
        codes, unique_values = pd.factorize(data[column])
        data[column] = codes
        encoded_mappings[column] = {'codes': codes, 'unique_values': unique_values}

    return data, encoded_mappings

st.set_page_config(layout="wide")

st.title('Hotel Bookings')
st.markdown(
    """
    Welcome to our app to predict the average daily rate of a hotel booking.  \n
    In the sidebar you can find different pages to explore the data, make predictions and the evaluations of the model. 
    
    ## Raw Data Source
    Check out this [Kaggle Page](https://www.kaggle.com/datasets/mojtaba142/hotel-booking) for the source of the raw data used in this app. 


"""
)

st.sidebar.markdown('# Hotel Bookings')
st.header('Loading data')


data_load_state = st.text('Loading raw data...')
df = load_csv_data('hotel_booking.csv')
data_load_state.text('Loading raw data...done!')

if st.checkbox("Show raw data first 5 rows"):
    st.write(df.head())

if 'df' not in st.session_state:
    st.session_state['df'] = df.copy()

data_load_state = st.text('Loading descriptive data...')
descriptive_data = data_preparation(df.copy())
data_load_state.text('Loading descriptive data...done!')

if 'descriptive_data' not in st.session_state:
    st.session_state['descriptive_data'] = descriptive_data

df = data_preparation(df)

st.header('Modeling Data')
data_load_state = st.text('Modeling data...')

#set categorical columns to numberical values
df, mappings = factorize_columns(df, ['hotel','meal','market_segment','distribution_channel','reserved_room_type','assigned_room_type', 'deposit_type', 'customer_type', 'reservation_status'])
st.session_state['mappings'] = mappings

#drop columns used in visualizations only
df.drop(['country','total_revenues','total_stay_in_nights'], axis=1, inplace = True)

y = df[['adr']]
X = df.drop(['adr', 'reservation_status_date','is_repeated_guest'],axis=1)

Xtrain, Xrest, ytrain, yrest = train_test_split(X, y, test_size=0.2)
Xval, Xtest, yval, ytest = train_test_split(Xrest, yrest, test_size=0.5)

##avoid dataleakage. filling in missing values after train test split
Xtest['children'].fillna(Xtest['children'].median(), inplace=True)
ytest['children'].fillna(ytest['children'].median(), inplace=True)
Xtest['children']=Xtest['children'].astype(int)
ytest['children']=ytest['children'].astype(int)


lr_model = joblib.load('models/lr_model.joblib')
if 'lr_model' not in st.session_state:
    st.session_state['lr_model'] = lr_model
xgb_model = joblib.load('models/xgb_model.joblib')
if 'xgb_model' not in st.session_state:
    st.session_state['xgb_model'] = xgb_model
knn_model = joblib.load('models/knn_model.joblib')
if 'knn_model' not in st.session_state:
    st.session_state['knn_model'] = knn_model

data_load_state.text('Modeling data...done!')
