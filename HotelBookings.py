import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    data = pd.read_csv("hotel_booking.csv")
    data.drop(['email', 'credit_card', 'phone-number', 'name'], axis=1, inplace = True)
    return data

data_load_state = st.text('Loading data...')
df = load_data()
data_load_state.text('Loading data...done!')

if st.checkbox("Show raw data first 5 rows"):
    st.write(df.head())

if 'df' not in st.session_state:
    st.session_state['df'] = df.copy()


#data cleaning for descriptive analysis on streamlit

df['arrival_date_month'] = pd.to_datetime(df.arrival_date_month, format='%B').dt.month
df['total_revenues'] = df['adr'] * (df['stays_in_weekend_nights'] + df['stays_in_week_nights'])
df['reservation_status_date']=df['reservation_status_date'].astype('datetime64[ns]')
df.drop(['agent', 'company'], axis=1, inplace = True)
df['children'].fillna(df['children'].median(), inplace=True)
df['children']=df['children'].astype(int)
mode_country=df['country'].mode()[0]
df['country'] = df['country'].fillna(mode_country)
df.drop(df['adr'].idxmax(), inplace = True)
df['total_stay_in_nights'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']


descriptive_data = df
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



data = df
data, mappings = factorize_columns(data, ['hotel','meal','market_segment','distribution_channel','reserved_room_type','assigned_room_type', 'deposit_type', 'customer_type', 'reservation_status'])
cancelled_percentage = data["is_canceled"].value_counts(normalize = True)
is_canceled_counts = data["is_canceled"].value_counts()
cancelled_data = data[data['is_canceled'] == 1]
top_10_countries_canceled = cancelled_data['country'].value_counts()[:10]
lost_revenues = cancelled_data['total_revenues'].sum()
top_10_countries_canceled_revenues = cancelled_data.groupby('country')['total_revenues'].sum()[:10]
current_revenues = data[data['is_canceled'] == 0]['total_revenues'].sum()
#data[data['is_canceled'] == 0]['is_repeated_guest'].sum()
data.drop(['country','total_revenues','total_stay_in_nights'], axis=1, inplace = True)



def RMSE(y,y_pred):
  mse=mean_squared_error(y, y_pred)
  return np.sqrt(mse)

y = data[['adr']]
X = data.drop(['adr', 'reservation_status_date'],axis=1)

Xtrain, Xrest, ytrain, yrest = train_test_split(X, y, test_size=0.2)
Xval, Xtest, yval, ytest = train_test_split(Xrest, yrest, test_size=0.5)

lr = LinearRegression()
lr.fit(Xtrain, ytrain)
#st.write(df.columns.tolist())

columns = Xtrain.columns.values.tolist()
coefs = lr.coef_.ravel().tolist()


X.drop(["is_repeated_guest"], axis=1, inplace = True)

Xtrain, Xrest, ytrain, yrest = train_test_split(X, y, test_size=0.2)
Xval, Xtest, yval, ytest = train_test_split(Xrest, yrest, test_size=0.5)
lr.fit(Xtrain, ytrain)

columns = Xtrain.columns.values.tolist()
coefs = lr.coef_.ravel().tolist()


lr.fit(Xtrain, ytrain)

def calculate_relative_error(rmse, y):
    y_range = y.max() - y.min()
    relative_error = float((rmse / y_range) * 100)

    return relative_error

y_pred_lr = lr.predict(Xtest)
rmse_adr = RMSE(yval, y_pred_lr)
relative_error = calculate_relative_error(rmse_adr, y)



kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = cross_val_score(lr, Xtrain, ytrain, cv=kf, scoring='neg_root_mean_squared_error')


RMSE(yval, y_pred_lr)

xgb = xgboost.XGBRegressor()
xgb.fit(Xtest, ytest)

y_pred_xgb = xgb.predict(Xtest)
rmse_adr = RMSE(yval, y_pred_xgb)
relative_error = calculate_relative_error(rmse_adr, y)



kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = cross_val_score(xgb, Xtrain, ytrain, cv=kf, scoring='neg_root_mean_squared_error')


RMSE(yval, y_pred_xgb)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(Xtrain, ytrain)

y_pred_knn = knn.predict(Xtest)
rmse_adr = RMSE(yval, y_pred_knn)
relative_error = calculate_relative_error(rmse_adr, y)



kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = cross_val_score(knn, Xtrain, ytrain, cv=kf, scoring='neg_root_mean_squared_error')



RMSE(yval, y_pred_knn)

st.session_state['kf_model'] = kf
st.session_state['lr_model'] = lr
st.session_state['knn_model'] = knn
