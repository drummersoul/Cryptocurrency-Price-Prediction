import os
import pandas as pd
import streamlit as st
from pathlib import Path
import pandas as pd


def get_csv_file(file_name):
    # Get the current working directory
    current_directory = os.getcwd()

    current_directory = current_directory.strip('/')
    file_name = file_name.strip('/')
    # Specify the relative path to the CSV file from the current directory
    print(current_directory)
    if current_directory.endswith("\src"):
        current_directory = current_directory[0:len(current_directory) - 4]
    csv_file_location = os.path.join(current_directory, "data", file_name)

    return csv_file_location


def get_data(file_name):
    current_script_path = Path(__file__).parent
    data_path = current_script_path.parent.parent / 'data' / file_name

    try:
        dataframe = pd.read_csv(data_path)
        print(f"Successfully loaded the file: {data_path}")
        return dataframe
    except FileNotFoundError as e:
        print(f"File not found at: {data_path}. Error: {e}")
        return None
    except Exception as e:
        print(f"An error occurred when attempting to read the file: {e}")
        return None


def get_specific_data(data_frame: pd.DataFrame, crypto_name: str):
    """return data for specific crypto"""
    return data_frame.loc[data_frame['crypto_name'] == crypto_name]


def display_kpis(train_acc, test_acc, knn_recall, knn_precision,model_name):
    kpi1, kpi2,kpi3,kpi4 = st.columns(4)
    theme = st.get_option("theme.base")
    font_color = "white" if theme == "dark" else "black"
    
    with kpi1:
        st.markdown("**Accuracy Testing**")
        st.markdown(f"<h1 style='text-align: left; color: {font_color};'>{round(test_acc,2)}</h1>", unsafe_allow_html=True)
    with kpi2:
        st.markdown("**Accuracy Training**")
        st.markdown(f"<h1 style='text-align: left; color: {font_color};'>{round(train_acc,2)}</h1>", unsafe_allow_html=True)
    with kpi3:
        st.markdown("**Recall**")
        st.markdown(f"<h1 style='text-align: left; color: {font_color};'>{round(knn_recall,2)}</h1>", unsafe_allow_html=True)
    with kpi4:
        st.markdown("**Precision**")
        st.markdown(f"<h1 style='text-align: left; color: {font_color};'>{round(knn_precision,2)}</h1>", unsafe_allow_html=True)

def setup_tabs():
    return st.tabs(["Historical_Data", "Dataset", "Heatmap", "Confusion Matrix", "ROC Curve"])