import datetime

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = st.session_state['df']
model_lr = st.session_state['lr_model']
model_kf = st.session_state['kf_model']
model_knn = st.session_state['knn_model']

def get_week_nights(start_date, stop_date):
    week_nights = 0
    current_date = start_date

    while current_date <= stop_date:
        if current_date.weekday() < 5:  # 0-4 denotes Monday-Friday
            week_nights += 1
        current_date += datetime.timedelta(days=1)

    return week_nights


def get_weekend_nigts(start_date, stop_date):
    weekend_nights = 0
    current_date = start_date
    while current_date <= stop_date:
        if current_date.weekday() >= 5:  # 5-6 denotes Saturday-Sunday
            weekend_nights += 1
        current_date += datetime.timedelta(days=1)
    return weekend_nights


def reformat_input(form):
    reformatted_data = {
        'hotel': [form.get('hotel')],
        'is_canceled': [form.get('is_canceled')],
        'lead_time': [form.get('lead_time')],
        'arrival_date_year': [form.get('arrival_date').year],
        'arrival_date_month': [form.get('arrival_date').month],
        'arrival_date_week_number': [form.get('arrival_date').isocalendar()[1]],
        'arrival_date_day_of_month': [form.get('arrival_date').day],
        'stays_in_weekend_nights': [get_weekend_nigts(form.get('arrival_date'), form.get('depart_date'))],
        'stays_in_week_nights': [get_week_nights(form.get('arrival_date'), form.get('depart_date'))],
        'adults': [form.get('adults')],
        'children': [form.get('children')],
        'babies': [form.get('babies')],
        'meal': [form.get('meal')],
        'market_segment': [form.get('market_segment')],
        'distribution_channel': [form.get('distribution_channel')],
        'previous_cancellations': [form.get('previous_cancellations')],
        'previous_bookings_not_canceled': [form.get('previous_bookings_not_canceled')],
        'reserved_room_type': [form.get('reserved_room_type')],
        'assigned_room_type': [form.get('assigned_room_type')],
        'booking_changes': [form.get('booking_changes')],
        'deposit_type': [form.get('deposit_type')],
        'days_in_waiting_list': [form.get('days_in_waiting_list')],
        'customer_type': [form.get('customer_type')],
        'required_car_parking_spaces': [form.get('required_car_parking_spaces')],
        'total_of_special_requests': [form.get('total_of_special_requests')],
        'reservation_status': [form.get('reservation_status')],
    }
    st.sidebar.write(pd.DataFrame(reformatted_data).info())
    return pd.DataFrame(reformatted_data)


def create_form():
    with st.sidebar.form(key='my_form'):
        st.markdown("# Fill in the form")

        # Create input elements with labels that match the data keys and default values from data
        raw_inputs = {
            "hotel": st.number_input("Hotel", step=1, min_value=0, max_value=2),
            "is_canceled": st.checkbox("Is canceled"),
            "lead_time": st.number_input("Lead time", step=1, min_value=0),
            "arrival_date": st.date_input('Arrival Date', datetime.date.today()),
            "depart_date": st.date_input('Departing date', datetime.date.today() + datetime.timedelta(days=5)),
            "adults": st.number_input("Amount of adults", step=1, min_value=0),
            "children": st.number_input("Amount of children", step=1, min_value=0),
            "babies": st.number_input("Amount of babies", step=1, min_value=0),
            "meal": st.number_input("Meal", step=1, min_value=0),
            "market_segment": st.number_input("Market segment", step=1, min_value=0),
            "distribution_channel": st.number_input("Distribution Channel", step=1, min_value=0),
            "previous_cancellations": st.number_input("Previous Cancellations", step=1, min_value=0),
            "previous_bookings_not_canceled": st.number_input("Previous bookings not canceled", step=1, min_value=0),
            "reserved_room_type": st.number_input("Room Type", step=1, min_value=0),
            "assigned_room_type": st.number_input("Room type assigned", step=1, min_value=0),
            "booking_changes": st.checkbox("Booking changes"),
            "deposit_type": st.number_input("Deposit type", step=1, min_value=0),
            "days_in_waiting_list": st.number_input("Days on waiting list", step=1, min_value=0),
            "customer_type": st.number_input("Customer type", step=1, min_value=0),
            "required_car_parking_spaces": st.number_input("Required parking spaces", step=1, min_value=0),
            "total_of_special_requests": st.number_input("Amount of special requests", step=1, min_value=0),
            "reservation_status": st.number_input("Reservation status", step=1, min_value=0),
        }

        if not raw_inputs.get('arrival_date') < raw_inputs.get('depart_date'):
            st.sidebar.error('Error: Departure date must fall after arrival date.')
        else:
            submitted = st.form_submit_button(label='Submit')

    if submitted:
        make_prediction(raw_inputs)


def make_prediction(form_value):
    prediction_lr = model_lr.predict(reformat_input(form_value))
    st.write(f'The linear regression predicted output is: {prediction_lr}')
    prediction_kf = model_lr.predict(reformat_input(form_value))
    st.write(f'The kf predicted output is: {prediction_kf}')
    prediction_knn = model_lr.predict(reformat_input(form_value))
    st.write(f'The knn predicted output is: {prediction_knn}')


if __name__ == "__main__":
    create_form()