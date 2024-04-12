from utils.utils import get_data, get_specific_data, display_kpis, setup_tabs
import streamlit as st
import numpy as np
from graphs import Graphs


class Dashboard:

    def __init__(self) -> None:
        self.graph = Graphs()

    def dashboard(self, logistic_data: tuple, xgb_data: tuple, rf_data: tuple, knn_data: tuple, dataframes: tuple):

        train_acc_logistic, test_acc_logistic, logistic_recall, logistic_precison, logistic_fig, fpr_logistic, \
            tpr_logistic, roc_auc_logistic = logistic_data
        train_acc_xgb, test_acc_xgb, xgbclassifier_recall, xgbclassifier_precison, xgbclassifier_fig, fpr_xgb, \
            tpr_xgb, roc_auc_xgb = xgb_data
        rf_train_acc, rf_test_acc, rf_recall, rf_precison, rf_fig, fpr_rf, tpr_rf, roc_auc_rf = rf_data
        train_acc_knn, test_acc_knn, knn_recall, knn_precison, knn_fig, fpr_knn, tpr_knn, roc_auc_knn = knn_data
        df, sub_df = dataframes

        # To decide if we want to show plots or not
        show_figure = True
        st.set_page_config(
            page_title="Crypto Prediciton Dashboard",
            page_icon="💸",
            layout="centered",
            initial_sidebar_state="expanded")
        # Title
        st.title("Litecoin Price Prediction Dashboard")

        opcion = st.sidebar.selectbox(
            'Select a model',
            ('Logistic Regression', 'XGB Classifier', 'Random Forest', "KNN", "More")  # available options
        )
        def model_specific_sidebar():
            start_date = st.sidebar.date_input("Start Date", min_value=df['date'].min().date(), max_value=df['date'].max().date(), value=df['date'].min().date())
            end_date = st.sidebar.date_input("End Date", min_value=df['date'].min().date(), max_value=df['date'].max().date(), value=df['date'].max().date())
            start_date = np.datetime64(start_date)
            end_date = np.datetime64(end_date)
            filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            return filtered_df
        # Content based on option selected by user
        if opcion == 'Logistic Regression':
            display_kpis(train_acc_logistic, test_acc_logistic, logistic_recall, logistic_precison,
                         "Logistic Regression")
            filtered_df = model_specific_sidebar()
            with st.container():
                # Pestañas para organizar contenido diferente
                tab1, tab2, tab3, tab4, tab5 = setup_tabs()
            with tab1:
                st.set_option('deprecation.showPyplotGlobalUse', False)   #Dashboard
                st.pyplot(self.graph.basicPlot(y = filtered_df['close'], title='Crypto Close Price', y_label= 'Price in Dollars', show_figure = show_figure))
            with tab2:
                st.dataframe(df)  # Dashboard
            with tab3:
                st.pyplot(self.graph.graphCorrelation(sub_df.iloc[:, 1:], "Correlation HeatMap for Litecoin",
                                                      show_figure=show_figure))
            with tab4:
                st.pyplot(logistic_fig)
            with tab5:
                st.pyplot(self.graph.roc_cure_for_one_model(fpr_logistic, tpr_logistic, roc_auc_logistic,
                                                            "Logistic Regression"))
        elif opcion == 'XGB Classifier':
            # Show KPI option 2
            display_kpis(train_acc_xgb, test_acc_xgb, xgbclassifier_recall, xgbclassifier_precison, "XGB Classifier")
            filtered_df = model_specific_sidebar()
            # Contenedor para más organización si es necesario
            with st.container():
                # Pestañas para organizar contenido diferente
                tab1, tab2, tab3, tab4, tab5 = setup_tabs()
            with tab1:
                st.set_option('deprecation.showPyplotGlobalUse', False)   #Dashboard
                st.pyplot(self.graph.basicPlot(y = filtered_df['close'], title='Crypto Close Price', y_label= 'Price in Dollars', show_figure = show_figure))
            with tab2:
                st.dataframe(df)  # Dashboard
            with tab3:
                st.pyplot(self.graph.graphCorrelation(sub_df.iloc[:, 1:], "Correlation HeatMap for Litecoin",
                                                      show_figure=show_figure))
            with tab4:
                st.pyplot(xgbclassifier_fig)
            with tab5:
                st.pyplot(self.graph.roc_cure_for_one_model(fpr_xgb, tpr_xgb, roc_auc_xgb, "XGBClassifier"))
        elif opcion == 'Random Forest':
            # Show KPI option 2
            display_kpis(rf_train_acc, rf_test_acc, rf_recall, rf_precison, "Random Forest")
            filtered_df = model_specific_sidebar()
            # Contenedor para más organización si es necesario
            with st.container():
                # Pestañas para organizar contenido diferente
                tab1, tab2, tab3, tab4, tab5 = setup_tabs()
            with tab1:
                st.set_option('deprecation.showPyplotGlobalUse', False)   #Dashboard
                st.pyplot(self.graph.basicPlot(y = filtered_df['close'], title='Crypto Close Price', y_label= 'Price in Dollars', show_figure = show_figure))
            with tab2:
                st.dataframe(df)
            with tab3:
                st.pyplot(self.graph.graphCorrelation(sub_df.iloc[:, 1:], "Correlation HeatMap for Litecoin",
                                                      show_figure=show_figure))
            with tab4:
                st.pyplot(rf_fig)
            with tab5:
                st.pyplot(self.graph.roc_cure_for_one_model(fpr_rf, tpr_rf, roc_auc_rf, "Random Forest"))
        elif opcion == 'KNN':
            # Show KPI option 2
            display_kpis(train_acc_knn, test_acc_knn, knn_recall, knn_precison, "KNN")
            filtered_df = model_specific_sidebar()
            # Contenedor para más organización si es necesario
            with st.container():
                # Pestañas para organizar contenido diferente   
                tab1, tab2, tab3, tab4, tab5 = setup_tabs()
            with tab1:
                st.set_option('deprecation.showPyplotGlobalUse', False)   #Dashboard
                st.pyplot(self.graph.basicPlot(y = filtered_df['close'], title='Crypto Close Price', y_label= 'Price in Dollars', show_figure = show_figure))
            with tab2:
                st.dataframe(df)
            with tab3:
                st.pyplot(self.graph.graphCorrelation(sub_df.iloc[:, 1:], "Correlation HeatMap for Litecoin",
                                                      show_figure=show_figure))
            with tab4:
                st.pyplot(knn_fig)
            with tab5:
                st.pyplot(self.graph.roc_cure_for_one_model(fpr_knn, tpr_knn, roc_auc_knn, "KNN"))

        elif opcion == "More":
            with st.container():
                # Pestañas para organizar contenido diferente
                tab1, tab2, tab3 = st.tabs(["Historical_Data", "Dataset", "Heatmap"])
            with tab1:
                st.set_option('deprecation.showPyplotGlobalUse', False)  # Dashboard
                st.pyplot(self.graph.basicPlot(y=df['close'], title='Crypto Close Price', y_label='Price in Dollars',
                                               show_figure=show_figure))
            with tab2:
                st.dataframe(df)  # Dashboard
            with tab3:
                st.pyplot(self.graph.graphCorrelation(sub_df.iloc[:, 1:], "Correlation HeatMap for Litecoin",
                                                      show_figure=show_figure))

