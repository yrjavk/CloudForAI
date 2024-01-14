from datetime import date

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = st.session_state['df']
model = st.session_state['lr_model']

data.describe()

cancelled_data = data[data['is_canceled'] == 1]
# top_10_countries_canceled = cancelled_data['country'].value_counts()[:10]
is_canceled_counts = data["is_canceled"].value_counts()
cancelled_percentage = data["is_canceled"].value_counts(normalize = True)
data['total_revenues'] = data['adr'] * (data['stays_in_weekend_nights'] + data['stays_in_week_nights'])
#lost_revenues = cancelled_data['total_revenues'].sum()
non_cancelled_data = data[data['is_canceled'] == 0]

with st.sidebar.form(key='input_form'):
    arrival_date = st.date_input('Select the arrival date', min_value=date(2000, 1, 1))
    hotel = st.radio('Select the type of hotel', options=[1, 2, 3])
    meal = st.selectbox('Select the type of meal', options=[True, False])
    market_segment = st.selectbox('Select the market_segment', options=[1, 2, 3, 4])

    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    st.write(f"You entered: ")
    st.write(f"arrival_date_year={arrival_date.year}")
    st.write(f"arrival_date_month={arrival_date.month}")
    # st.write(f"arrival_date_week_number={arrival_date.}")
    st.write(f"hotel={hotel}")
    st.write(f"meal={meal}")
    st.write(f"market_segment={market_segment}")

    # include similar entries for the rest of the inputs...


data = {
    'hotel': [1],
    'is_canceled': [0],
    'lead_time': [130],
    'arrival_date_year': [2016],
    'arrival_date_month': [5],
    'arrival_date_week_number': [20],
    'arrival_date_day_of_month': [14],
    'stays_in_weekend_nights': [1],
    'stays_in_week_nights': [5],
    'adults': [2],
    'children': [1],
    'babies': [0],
    'meal': [0],
    'market_segment': [2],
    'distribution_channel': [1],
    'previous_cancellations': [0],
    'previous_bookings_not_canceled': [0],
    'reserved_room_type': [0],
    'assigned_room_type': [0],
    'booking_changes': [0],
    'deposit_type': [0],
    'days_in_waiting_list': [10],
    'customer_type': [2],
    'required_car_parking_spaces': [1],
    'total_of_special_requests': [0],
    'reservation_status': [1],
}
def make_dataframe():

    return pd.DataFrame(data)

model = st.session_state['lr_model']
prediction = model.predict(make_dataframe())
st.write(f'The predicted output is: {prediction}')
