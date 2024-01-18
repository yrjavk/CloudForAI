import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
import joblib
from utils import RMSE, calculate_relative_error
from streamlit_extras.switch_page_button import switch_page

def load_models():
    models = {
        "lr_model": joblib.load('models/lr_model.joblib'),
        "xgb_model": joblib.load('models/xgb_model.joblib'),
        "knn_model": joblib.load('models/knn_model.joblib'),
    }
    for model_name, model in models.items():
        if str(model_name) not in st.session_state:
            switch_page("Hotel Bookings")
    

def process_in_batches(model, model_name, Xtest):
    batch_size = 100
    model_pred = []

    data_load_state = st.text(f"Processing data for {model_name}")
    start_time = time.time()

    for i in range(0, len(Xtest), batch_size):
        remaining = f"{model_name}: {len(Xtest) - i} batches remaining"
        data_load_state.text(remaining)

        current_batch = Xtest.iloc[i:i + batch_size, :]
        current_predictions = model.predict(current_batch)
        model_pred.extend(current_predictions)

    end_time = time.time()
    elapsed_time = end_time - start_time
    data_load_state.text(f"Elapsed Time for {model_name}: {elapsed_time:.2f} seconds")

    return np.array(model_pred)

def plot_predictions(models, rmse_values, relative_error_values):
    fig, pos = plt.subplots(1, 2, figsize=(15, 5))

    pos[0].bar(models, rmse_values)
    pos[0].set_title("RMSE")
    pos[0].set_ylabel("RMSE")

    pos[1].bar(models, relative_error_values)
    pos[1].set_title("Relative Error")
    pos[1].set_ylabel("Relative Error (%)")

    st.pyplot(fig)

def main():

    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    load_models()

    st.markdown("# Evaluations")
    st.sidebar.markdown("# Evaluations")

    lr_model = st.session_state['lr_model']
    xgb_model = st.session_state['xgb_model']
    knn_model = st.session_state['knn_model']

    models = [("Linear Regression", lr_model), ("XGBoost", xgb_model), ("KNN", knn_model)]

    Xtest = st.session_state.get('Xtest')
    ytest = st.session_state.get('ytest')
    y_range = st.session_state.get('y_range')

    log.info(f"X input type: {type(Xtest)}\Size: {Xtest.shape}")
    log.info(f"Y input type: {type(ytest)}\Size: {ytest.shape}")
    log.info(f"Target value range: {y_range}")

    predictions_list = []

    for model_name, model in models:
        model_pred = process_in_batches(model, model_name, Xtest)
        predictions_list.append(model_pred)

    rmse_values = [RMSE(ytest, pred) for pred in predictions_list]
    relative_error_values = [calculate_relative_error(rmse, y_range) for rmse in rmse_values]
    model_names = [model[0] for model in models]

    plot_predictions(model_names, rmse_values, relative_error_values)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Root Mean Squared Error")
        for model_name, rmse in zip(model_names, rmse_values):
            st.write(f"{model_name}: {rmse}")

    with col2:
        st.subheader("Relative Error")
        for model_name, relative_error in zip(model_names, relative_error_values):
            st.write(f"{model_name}: {relative_error}%")

if __name__ == "__main__":
    main()
