import os
import pandas as pd
import streamlit as st
from models.ml_models import Models

def get_csv_file(file_name):
    
    # Get the current working directory
    current_directory = os.getcwd()

    current_directory = current_directory.strip('/')
    file_name = file_name.strip('/') 
    # Specify the relative path to the CSV file from the current directory
    print(current_directory)
    if(current_directory.endswith("\src")):
        current_directory = current_directory[0:len(current_directory)-4]
    csv_file_location = os.path.join(current_directory, "data", file_name)

    return csv_file_location

def get_data(file_name):
    """return data as data frame"""
    path = get_csv_file(file_name)

    # Check if the file exists
    if os.path.exists(path):
        dataframe = pd.read_csv(path)
        print("File successfully loaded.")
        return dataframe
    else:
        print(f"File not found at location: {path}")

def get_specific_data(data_frame: pd.DataFrame, crypto_name:str):
    """return data for specific crypto"""
    return data_frame.loc[data_frame['crypto_name'] == crypto_name]

def display_kpis(train_acc, test_acc, knn_recall, knn_precision,model_name):
    kpi1, kpi2,kpi3,kpi4 = st.columns(4)
    with kpi1:
        st.markdown("**Accuracy Testing**")
        st.markdown(f"<h1 style='text-align: left; color: white;'>{round(test_acc,2)}</h1>", unsafe_allow_html=True)
    with kpi2:
        st.markdown("**Accuracy Training**")
        st.markdown(f"<h1 style='text-align: left; color: white;'>{round(train_acc,2)}</h1>", unsafe_allow_html=True)
    with kpi3:
        st.markdown("**Recall**")
        st.markdown(f"<h1 style='text-align: left; color: white;'>{round(knn_recall,2)}</h1>", unsafe_allow_html=True)
    with kpi4:
        st.markdown("**Precision**")
        st.markdown(f"<h1 style='text-align: left; color: white;'>{round(knn_precision,2)}</h1>", unsafe_allow_html=True)

def setup_tabs():
    return st.tabs(["Historical_Data", "Dataset","Heatmap", "Confusion Matrix", "ROC Curve"])