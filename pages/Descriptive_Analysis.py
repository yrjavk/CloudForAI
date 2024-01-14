import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = st.session_state['df']

cancelled_data = data[data['is_canceled'] == 1]
top_10_countries_canceled = cancelled_data['country'].value_counts()[:10]
is_canceled_counts = data["is_canceled"].value_counts()
cancelled_percentage = data["is_canceled"].value_counts(normalize = True)
data['total_revenues'] = data['adr'] * (data['stays_in_weekend_nights'] + data['stays_in_week_nights'])
#lost_revenues = cancelled_data['total_revenues'].sum()
non_cancelled_data = data[data['is_canceled'] == 0]


st.markdown("# Descriptive Analysis")
st.sidebar.markdown("# Descriptive Analysis")

tab1, tab2 = st.tabs(["Data Info", "Graphs"])

with tab1:
    if data is not None:
        st.header("Meta-data")
        row_count = data.shape[0]
        col_count = data.shape[1]
        
        missing_value_row_count = data[data.isna().any(axis=1)].shape[0]

        table_description=f"""
        |Description | Value |
        | --- | --- |
        | # of rows | {row_count} |
        |# of columns | {col_count}|
        |# of rows with missing values |{missing_value_row_count}|"""
        
        st.markdown(table_description)

        st.header("Columns")
        columns = list(data.columns)
        column_info_table = pd.DataFrame({
            "Column": columns,
            "Data_type": data.dtypes.tolist()
        })
        st.dataframe(column_info_table, hide_index=True)


        st.header("Missing Values")
        st.text("")
        st.write(data.isna().sum().sort_values(ascending=False))




with tab2:
    if data is not None:

        st.line_chart(data,)



        st.subheader('Top 10 countries with cancellations in %')
        fig = plt.figure(figsize = (6,6))
        plt.pie(top_10_countries_canceled,autopct = "%.2f",labels = top_10_countries_canceled.index)
        st.pyplot(fig)


        st.subheader('Top 10 countries revenue lost due to cancellation in %')
        top_10_countries_canceled_revenues = cancelled_data.groupby('country')['total_revenues'].sum()[:10]
        fig = plt.figure(figsize = (6,6))
        plt.pie(top_10_countries_canceled_revenues,autopct = "%.2f",labels = top_10_countries_canceled_revenues.index)
        st.pyplot(fig)

        st.subheader('Average daily price')
        fig = plt.figure(figsize=(6,4))
        sns.lineplot(x='arrival_date_month', y='adr', hue='hotel', data=data)
        plt.ylabel("Average daily price")
        plt.xlabel("Months")
        p = plt.xticks(rotation=30)
        st.pyplot(fig)


