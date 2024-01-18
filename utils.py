import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def load_csv_data(filename):
    data = pd.read_csv(filename)
    data.drop(['email', 'credit_card', 'phone-number', 'name'], axis=1, inplace = True)
    return data

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

def factorize_columns(data, columns_to_factorize):
    encoded_mappings = {}

    for column in columns_to_factorize:
        codes, unique_values = pd.factorize(data[column])
        data[column] = codes
        encoded_mappings[column] = {'codes': codes, 'unique_values': unique_values}

    return data, encoded_mappings

def RMSE(y,y_pred):
  mse=mean_squared_error(y, y_pred)
  return np.sqrt(mse)

def calculate_relative_error(rmse, range):
    relative_error = float((rmse/range.item()) * 100)

    return relative_error