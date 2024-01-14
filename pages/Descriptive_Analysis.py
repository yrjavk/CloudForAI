import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




rawdata = st.session_state['df']
st.markdown("# Descriptive Analysis")
st.sidebar.markdown("# Descriptive Analysis")

tab1, tab2 = st.tabs(["Data Info", "Graphs"])

with tab1:
    if rawdata is not None:
        st.header("Meta-data")
        row_count = rawdata.shape[0]
        col_count = rawdata.shape[1]
        
        missing_value_row_count = rawdata[rawdata.isna().any(axis=1)].shape[0]

        table_description=f"""
        |Description | Value |
        | --- | --- |
        | # of rows | {row_count} |
        |# of columns | {col_count}|
        |# of rows with missing values |{missing_value_row_count}|"""
        
        st.markdown(table_description)

        st.header("Columns")
        columns = list(rawdata.columns)
        column_info_table = pd.DataFrame({
            "Column": columns,
            "Data_type": rawdata.dtypes.tolist()
        })
        st.dataframe(column_info_table, hide_index=True)

        st.header("Missing Values")
        st.text("")
        st.write(rawdata.isna().sum().sort_values(ascending=False))


descriptive_data = st.session_state['descriptive_data']
cancelled_data = descriptive_data[descriptive_data['is_canceled'] == 1]
is_canceled_counts = descriptive_data["is_canceled"].value_counts()
cancelled_percentage = descriptive_data["is_canceled"].value_counts(normalize = True)
lost_revenues = cancelled_data['total_revenues'].sum()
non_cancelled_data = descriptive_data[descriptive_data['is_canceled'] == 0]
top_10_countries_canceled = cancelled_data['country'].value_counts()[:10]
top_10_countries_canceled_revenues = cancelled_data.groupby('country')['total_revenues'].sum()[:10]


with tab2:
    if descriptive_data is not None:

        # Line chart
        st.subheader('Average Daily Rate (ADR) by Month and Hotel')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(x='arrival_date_month', y='adr', hue='hotel', data=descriptive_data, ax=ax)
        plt.ylabel("Average Daily Rate (ADR)")
        plt.xlabel("Month")
        plt.title("ADR by Month and Hotel")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

        #bar chart
        st.subheader('Cancellation Count')
        st.bar_chart(descriptive_data['is_canceled'].value_counts())

        #pie chart
        st.subheader('Top 10 countries with cancellations in %')
        fig = plt.figure(figsize = (6,6))
        plt.pie(top_10_countries_canceled,autopct = "%.2f",labels = top_10_countries_canceled.index)
        st.pyplot(fig)

        #pie chart
        st.subheader('Top 10 countries revenue lost due to cancellation in %')
        top_10_countries_canceled_revenues = cancelled_data.groupby('country')['total_revenues'].sum()[:10]
        fig = plt.figure(figsize = (6,6))
        plt.pie(top_10_countries_canceled_revenues,autopct = "%.2f",labels = top_10_countries_canceled_revenues.index)
        st.pyplot(fig)



