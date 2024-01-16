import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_extras.switch_page_button import switch_page


if 'df' not in st.session_state:
    switch_page("Hotel Bookings")
if 'descriptive_data' not in st.session_state:
    switch_page("Hotel Bookings")

@st.cache_data
def load_data(data):
    return st.session_state[data]

@st.cache_data
def create_line_chart_adr(data):
        st.subheader('Average Daily Rate (ADR) by Month and Hotel')
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.set_facecolor('lightgrey')
        sns.lineplot(x='arrival_date_month', y='adr', hue='hotel', data=data, ax=ax)
        plt.ylabel("Average Daily Rate (ADR)")
        plt.xlabel("Month")
        plt.title("ADR by Month and Hotel")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

# Function to create a bar chart using Plotly
@st.cache_data
def create_cancellation_bar_chart(data):
    st.subheader('Cancellation Count')
    st.bar_chart(data['is_canceled'].value_counts())

# Function to create a pie chart using Plotly
@st.cache_data
def create_pie_chart(data, title):
    st.subheader(title)
    fig = plt.figure(figsize=(6, 6))
    fig.set_facecolor('lightgrey')
    plt.pie(data, autopct="%.2f", labels=data.index)
    st.pyplot(fig)

rawdata = load_data('df')
descriptive_data = load_data('descriptive_data')


st.markdown("# Descriptive Analysis")
st.sidebar.markdown("# Descriptive Analysis")

# Create a selectbox for hotel in the sidebar
all_hotels = descriptive_data['hotel'].unique()
selected_hotel = st.sidebar.multiselect('Hotel:', all_hotels, default=all_hotels)

# Subset of data based on the selection
subset_raw = rawdata[rawdata["hotel"].isin(selected_hotel)]
subset_desc = descriptive_data[descriptive_data["hotel"].isin(selected_hotel)]


#some calculations
cancelled_data = subset_desc[subset_desc['is_canceled'] == 1]
lost_revenues = cancelled_data['total_revenues'].sum()
top_10_countries_canceled = cancelled_data['country'].value_counts()[:10]
current_revenues = subset_desc[subset_desc['is_canceled'] == 0]['total_revenues'].sum()
top_10_countries_canceled_revenues = cancelled_data.groupby('country')['total_revenues'].sum()[:10]
op_10_countries_canceled_revenues = top_10_countries_canceled_revenues.sort_values(ascending=False)
missing_value_row_count = subset_raw.isna().any(axis=1).shape[0]


tab1, tab2 = st.tabs(["Data Info", "Graphs"])
with tab1:
    if rawdata is not None:

        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("Meta-data")
            row_count = subset_raw .shape[0]
            col_count = rawdata.shape[1]
        
            table_description=f"""
            |Description | Value |
            | --- | --- |
            | # of rows | {row_count} |
            |# of columns | {col_count}|
            |# of rows with missing values |{missing_value_row_count}|"""
            st.markdown(table_description)

        with col2:
            st.header("Columns")
            columns = list(rawdata.columns)
            column_info_table = pd.DataFrame({
                "Column": columns,
                "Data_type": rawdata.dtypes.tolist()
            })
            st.dataframe(column_info_table, hide_index=True)

        with col3:
            st.header("Missing Values")
            st.text("")
            st.write(subset_raw.isna().sum().sort_values(ascending=False))


with tab2:
    if descriptive_data is not None:

        st.write('Lost revenue due to cancellations: ', lost_revenues)
        st.write('Total revenue non cancelled entries: ', current_revenues)

        col1, col2 = st.columns(2)
        with col1:
            create_line_chart_adr(subset_desc)

        with col2:
            create_cancellation_bar_chart(subset_desc)

        col1, col2 = st.columns(2)
        with col1:
            create_pie_chart(top_10_countries_canceled, 'Top 10 countries with cancellations in %')

        with col2:
            create_pie_chart(top_10_countries_canceled_revenues, 'Top 10 countries revenue lost due to cancellation in %')

      

