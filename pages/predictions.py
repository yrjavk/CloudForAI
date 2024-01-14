from datetime import date

import streamlit as st
import pandas as pd

with st.sidebar.form(key='input_form'):
    arrival_date = st.date_input('Select the arrival date', min_value=date(2000, 1, 1))
    hotel = st.radio('Select the type of hotel', options=[1, 2, 3])
    meal = st.selectbox('Select the type of meal', options=[True, False])
    market_segment = st.selectbox('Select the market_segment', options=[1, 2, 3, 4])

    # include similar entries for the rest of the inputs...

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


# def make_dataframe():
#     data = {
#         'arrival_date_year': [2016],
#         'arrival_date_month': [5],
#         'arrival_date_week_number': [20],
#         'hotel': [1],
#         'meal': [0],
#         'market_segment': [2],
#         'distribution_channel': [1],
#         'reserved_room_type': [0],
#         'assigned_room_type': [0],
#         'deposit_type': [0],
#         'customer_type': [2],
#         'reservation_status': [1],
#         'is_canceled': [0],
#         'lead_time': [130],
#     }
#
#     return pd.DataFrame(data)
#
# new_data_predictions_knn = knn(make_dataframe())
#
# print(f"The predicted average daily revenue (adr) is: {new_data_predictions_knn[0]}")
