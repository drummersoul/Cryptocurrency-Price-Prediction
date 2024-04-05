import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import get_data, get_specific_data
import warnings  # Adding warning ignore to avoid issues with distplot
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split,TimeSeriesSplit,cross_val_score
from xgboost import XGBClassifier
from graphs import Graphs
from models.ml_models import Models
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from prophet.plot import (plot_plotly, 
                            plot_components_plotly,
                            plot_forecast_component)
from prophet.plot import plot_forecast_component
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings('ignore')
from sklearn.exceptions import ConvergenceWarning
# This will ignore all convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class DataUnderstanding:

    def __init__(self) -> None:
        pass

    def data_understanding(self, file_name : str):
        # Read dataset and display basic informations
        df = get_data(file_name)

        #instantiating Graph & Model classes
        graph = Graphs()
        model = Models()

        # Filtering data for only Litecoin
        df = get_specific_data(df, 'Litecoin')
        print(df.shape)

        # Removing columns we wont use because they have only null values
        df = df.drop(columns=["volume"])

        # Conver date object type to date type
        df['date'] = pd.to_datetime(df['date'])

        #Created a new dataframe to store newly create date column
        new_df=pd.DataFrame()
        new_df['m_date']=df['date']
        
        #In this section we will check for the number of missing dates
        new_df =new_df.set_index('m_date')
        
        #Create a full date range from the start to the end of your dataset
        date_range=pd.date_range(start=new_df.index.min(),end=new_df.index.max())
        
        #Identify missing dates by finding those which are not in your Datafreme's index
        missing_dates=date_range.difference(new_df.index)
        print(f'Missing Date:{missing_dates}')
        print(f'Missing Date Size{missing_dates.size}')
        
        #On this other section we will use linear interpolation t add the missing dates and values for each column
        # Set the 'date' column as the index.
        df.set_index('date', inplace=True)
        # Sort the Dataframe by the index.
        df.sort_index(inplace=True)
        # Create a full range of dates from the minimum to the maximum date.
        full_range = pd.date_range(start=df.index.min(), end=df.index.max())
        # Reindex the Dataframe to include all dates in the full range.
        df = df.reindex(full_range)
        # Select columns to interpolate.
        columns_to_interpolate = df.columns.difference([])
        # Interpolate only the selected columns.
        df[columns_to_interpolate] = df[columns_to_interpolate].interpolate(method='linear')
        # Reset the index to turn the 'date' index back into a column.
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'date'}, inplace=True)
        # Fill missing values in 'crypto_name' column with 'Litecoin'.
        df['crypto_name'] = df['crypto_name'].fillna('Litecoin')
        df.drop(['timestamp'], axis=1, inplace=True)
        print(df.isnull().sum())
        #To decide if we want to show plots or not
        show_figure = False

        # Plot historical close price
        graph.basicPlot(y = df['close'], title='Crypto Close Price',x_label= "years", y_label= 'Price in Dollars', show_figure = show_figure)

        # Define features for future use
        features = ['open', 'high', 'low', 'close']
        graph.distPlotWithSubPlot(df, features = features, rows = 2, cols = 2, show_figure = show_figure)

        # Extract the year from the 'date' column using the dt accessor in pandas
        df['year'] = df['date'].dt.year
        # Extract the month from the 'date' column using the dt accessor in pandas
        df['month'] = df['date'].dt.month
        # Extract the day from the 'date' column using the dt accessor in pandas
        df['day'] = df['date'].dt.day
        #new features derived using existing features
        df['open_close'] = df['open'] - df['close']
        df['low_high'] = df['low'] - df['high']
        df['MA7'] = df['close'].rolling(window=7).mean()
        df['MA30'] = df['close'].rolling(window=30).mean()
        df['Price_Change'] = df['close'].diff()
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)

        # Print the first few rows of the DataFrame to see the changes
        # print(df.head())

        # Group the DataFrame 'df' by the 'year' column and calculate the mean of each numeric column for each group
        # numeric_only is to calculate mean only for numbers
        data_grouped = df.groupby(by=['year']).mean(numeric_only=True)
        #print(data_grouped)

        bar_plot_features = ['open', 'high', 'low', 'close']
        # Plot a bar chart for the current column using the grouped data
        graph.barplotWithSubplot(data_grouped, bar_plot_features, 2, 2,show_figure = show_figure)

        # Keeping the columns for heatmap exploration
        sub_df = df[['open', 'high', 'low', 'close', 'marketCap', 'open_close', 'low_high', 'year', 'month', 'day', 'target']]
        sub_df.head()

        # Correlation for Litecoin crypto
        graph.graphCorrelation(sub_df.iloc[:, 1:], "Correlation HeatMap for Litecoin",show_figure = show_figure)

        visualize_cols = ['open', 'high', 'low', 'marketCap']

        # ploting graph to check correlation
        graph.scatterPlotWithSubPlot(sub_df, 'close', visualize_cols, 2, 2,show_figure = show_figure)

        # boxplot to check outliers with whisker_length(whis) of 1.5(default value)
        graph.boxPlotWithSubplot(sub_df, visualize_cols, 2, 2,show_figure = show_figure)
        
        #feature and target variables for classification
        X = df[['open', 'high', 'low', 'marketCap', 'year', 'month', 'day']]
        y = df['close']

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2022)

        #calculate the index for the split
        splitting_index = int(len(df) * 0.8)

        #split data without shuffling
        X_train = X[:splitting_index]
        X_test = X[splitting_index:]
        y_train = y[:splitting_index]
        y_test = y[splitting_index:]

        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_test: {y_test.shape}")

        y_train_class = (y_train.shift(-1) > y_train).astype(int)
        y_test_class = (y_test.shift(-1) > y_test).astype(int)

        #testing for class counts
        print("class counts of train_data:")
        print(y_train_class.value_counts()) 

        #LogisticRegression
        logistic_reg = model.logistic_regression(X_train, X_test, y_train_class, y_test_class)
        tscv = TimeSeriesSplit(n_splits=5)
        logistic_reg_model = LogisticRegression()
        # cv_scores = cross_val_score(logistic_reg_model, X_train, y_train_class, cv=tscv)
        # # Print cross-validation scores
        # print("Cross-validation scores LOG_REG:", cv_scores)
        # # Print mean and standard deviation of cross-validation scores
        # print("Mean CV score LOG_REG:", cv_scores.mean())
        # print("Standard deviation of CV scores LOG_REG:", cv_scores.std())

        train_acc_logistic, test_acc_logistic, y_pred_proba_logistic = logistic_reg
        print(f'Accuracy of Training: {train_acc_logistic}')
        print(f'Accuracy of Testing: {test_acc_logistic}')

        # model.xgb_gcv(X_train, X_test, y_train_class, y_test_class)
        # model.logr_gcv(X_train, X_test, y_train_class, y_test_class)

        #use XGBClassifier to tain a model and predict classes

        reg_lambda = 1.0
        reg_alpha = 0.5
        learning_rate = 0.01
        max_depth = 3
        xgbclassifier = model.xgbclassifier(reg_lambda, reg_alpha, learning_rate, max_depth, X_train, X_test, y_train_class, y_test_class)
        train_acc_xgb, test_acc_xgb, y_pred_proba_xgb = xgbclassifier
        tscv = TimeSeriesSplit(n_splits=5)
        xgb_cv_model = XGBClassifier()
        # cv_scores_xgb = cross_val_score(xgb_cv_model, X_train, y_train_class, cv=tscv)
        # # Print cross-validation scores
        # print("Cross-validation scores XGB:", cv_scores_xgb)
        # # Print mean and standard deviation of cross-validation scores
        # print("Mean CV score XGB:", cv_scores_xgb.mean())
        # print("Standard deviation of CV scores XGB:", cv_scores_xgb.std())
        print("Evaluation results for XGBClassifier:")
        print(f"training set accuracy: {train_acc_xgb}")
        print(f"test set accuracy: {test_acc_xgb}")

        prophet_data = df[['date','close']].copy()
        prophet_data.columns = ['ds', 'y']
        
        #Rename columns to 'ds' and 'y'
        prophet_model = Prophet()
        prophet_model.add_country_holidays(country_name='US')
        prophet_model.fit(prophet_data)

        future = prophet_model.make_future_dataframe(periods=365)
        forecast = prophet_model.predict(future)
        future = prophet_model.make_future_dataframe(periods=3, freq='M')
        fig1 = prophet_model.plot(forecast)
        plt.show()
        fig2 = prophet_model.plot_components(forecast)
        plt.show()
        
        #Time series forecasting after splitting the data
        split_date = '2021-04-01'
        train_df = df[df['date'] <= split_date].copy()
        test_df = df[df['date'] > split_date].copy()
        train_df.rename(columns={'date': 'ds', 'close': 'y'}, inplace=True)
        test_df.rename(columns={'date': 'ds', 'close': 'y'}, inplace=True)
        
        prophet_model1 = Prophet()
        prophet_model1.fit(train_df)
        forecast_on_test = prophet_model1.predict(test_df[['ds']])
        
        #Plot of the forecast with actual data
        graph.forecast_with_actual_data(prophet_model1, test_df, forecast_on_test,show_figure = show_figure)
        
        #Calculating MAE(Mean Absolute Error)
        mae_accu = mean_absolute_error(y_true=test_df['y'], y_pred=forecast_on_test['yhat'])
        print(f'Mean Absolute Error For Time series:{mae_accu}')

        # Calculating ROC curve and AUC for Logistic
        fpr_logistic, tpr_logistic, thresholds_logistic = roc_curve(y_test_class, y_pred_proba_logistic)
        roc_auc_logistic = roc_auc_score(y_test_class, y_pred_proba_logistic)

        # Calculating ROC curve and AUC for XGBoost
        fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test_class, y_pred_proba_xgb)
        roc_auc_xgb = roc_auc_score(y_test_class, y_pred_proba_xgb)

        # Plotting ROC curve for both models
        graph.roc_cure(fpr_logistic, tpr_logistic,roc_auc_logistic, fpr_xgb, tpr_xgb, roc_auc_xgb,show_figure = show_figure)

        #Random Forest Algorithem
        train_acc, test_acc, rf_y_pred = model.random_forest(X_train, X_test, y_train_class, y_test_class)
        print(" Random Forest -- Train Accuracy ==>> ", train_acc)
        print(" Random Forest -- Test Accuracy ==>> ", test_acc)

        #ploting Roc-curve for Random Forest
        fpr_rf, tpr_rf, threshold_rf = roc_curve(y_test_class, rf_y_pred)
        
        #Area under curve (auc)
        roc_auc_rf = roc_auc_score(y_test_class, rf_y_pred)
        graph.roc_cure_for_one_model(fpr_rf, tpr_rf, roc_auc_rf)

        print()
        print("KNN Model : ", end="\n\n")
        # Instantiate KNN model
        knn_model = KNeighborsClassifier(n_neighbors=5)

        # Train the model
        knn_model.fit(X_train, y_train_class)

        # Predict classes for test data
        y_pred_knn = knn_model.predict(X_test)

        # Evaluate model performance
        train_acc_knn = knn_model.score(X_train, y_train_class)
        test_acc_knn = knn_model.score(X_test, y_test_class)

        print("KNN - Train Accuracy:", train_acc_knn)
        print("KNN - Test Accuracy:", test_acc_knn)

         # Calculate ROC curve and AUC for KNN
        fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test_class, y_pred_knn)
        roc_auc_knn = roc_auc_score(y_test_class, y_pred_knn)

        # Plot ROC curve for KNN
        graph.roc_cure_for_one_model(fpr_knn, tpr_knn, roc_auc_knn)

        #To decide if we want to show plots or not
        show_figure = True   
        st.set_page_config(
            page_title="Crypto Prediciton Dashboard",
            page_icon="💸",
            layout="centered",
            initial_sidebar_state="expanded")
        
        opcion = st.sidebar.selectbox(
                'Select a model',
                ('Logistic Regression', 'XGBClassifier')  # available options
                )

        # Content based on option selected by user
        if opcion == 'Logistic Regression':
            # KPIS for option 1
            kpi1, kpi2 = st.columns(2)
            with kpi1:
                st.markdown("**Accuracy Testing**")
                st.markdown(f"<h1 style='text-align: left; color: black;'>{round(train_acc_logistic,2)}</h1>", unsafe_allow_html=True)
            with kpi2:
                st.markdown("**Accuracy Training**")
                st.markdown(f"<h1 style='text-align: left; color: black;'>{round(test_acc_logistic,2)}</h1>", unsafe_allow_html=True)
        elif opcion == 'XGBClassifier':
            # Show KPI option 2
            kpi1, kpi2 = st.columns(2)
            with kpi1:
                st.markdown("**Accuracy Testing**")
                st.markdown(f"<h1 style='text-align: left; color: black;'>{round(train_acc_xgb,2)}</h1>", unsafe_allow_html=True)
            with kpi2:
                st.markdown("**Accuracy Training**")
                st.markdown(f"<h1 style='text-align: left; color: black;'>{round(test_acc_xgb,2)}</h1>", unsafe_allow_html=True)