import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from utils.utils import get_data, get_specific_data
import warnings  # Adding warning ignore to avoid issues with distplot
import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from graphs import Graphs
from Ml_Models import Models
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

class DataUnderstanding:

    def __init__(self) -> None:
        pass

    def data_understanding(self, file_name : str):
        # Read dataset and display basic information
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

        # Plot historical close price
        graph.basicPlot(y = df['close'], title='Crypto Close Price', y_label= 'Price in Dollars')

        # Define features for future use
        features = ['open', 'high', 'low', 'close']
        graph.distPlotWithSubPlot(df, features = features, rows = 2, cols = 2)

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
        print(df.head())

        # Group the DataFrame 'df' by the 'year' column and calculate the mean of each numeric column for each group
        # numeric_only is to calculate mean only for numbers
        data_grouped = df.groupby(by=['year']).mean(numeric_only=True)
        print(data_grouped)

        bar_plot_features = ['open', 'high', 'low', 'close']
        # Plot a bar chart for the current column using the grouped data
        graph.barplotWithSubplot(data_grouped, bar_plot_features, 2, 2)

        # Keeping the columns for heatmap exploration
        sub_df = df[['open', 'high', 'low', 'close', 'marketCap', 'open_close', 'low_high', 'year', 'month', 'day', 'target']]
        sub_df.head()

        # Correlation for Litecoin crypto
        graph.graphCorrelation(sub_df.iloc[:, 1:], "Correlation HeatMap for Litecoin")

        visualize_cols = ['open', 'high', 'low', 'marketCap']

        # ploting graph to check correlation
        graph.scatterPlotWithSubPlot(sub_df, 'close', visualize_cols, 2, 2)

        # boxplot to check outliers with whisker_length(whis) of 1.5(default value)
        graph.boxPlotWithSubplot(sub_df, visualize_cols, 2, 2)
        
        #feature and target variables for classification
        X = df[['open', 'high', 'low', 'marketCap', 'year', 'month', 'day']]
        y = df['close']

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2022)

        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_test: {y_test.shape}")

        y_train_class = (y_train.shift(-1) > y_train).astype(int)
        y_test_class = (y_test.shift(-1) > y_test).astype(int)

        #LogisticRegression
        logistic_reg = model.logistic_regression(X_train, X_test, y_train_class, y_test_class)

        trained_logistic_model, train_acc, test_acc, y_pred_proba_logistic = logistic_reg
        print(f'Accuracy of Training: {train_acc}')
        print(f'Accuracy of Testing: {test_acc}')

        #display confusion matrix, and classification report
        model.display_classificiation_metrics(trained_logistic_model, X_test, y_test_class)

        #use XGBClassifier to tain a model and predict classes

        reg_lambda = 1.0
        reg_alpha = 0.5
        learning_rate = 0.01
        max_depth = 3
        xgbclassifier = model.xgbclassifier(reg_lambda, reg_alpha, learning_rate, max_depth, X_train, X_test, y_train_class, y_test_class)
        train_acc_xgb, test_acc_xgb, y_pred_proba_xgb = xgbclassifier

        print("Evaluation results for XGBClassifier:")
        print(f"training set accuracy: {train_acc_xgb}")
        print(f"test set accuracy: {test_acc_xgb}")

        prophet_data = df[['date','close']].copy()
        prophet_data.columns = ['ds', 'y']

        #Rename columns to 'ds' and 'y'

        prophet_model = Prophet()
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
        
        model = Prophet()
        model.fit(train_df)
        forecast_on_test=model.predict(test_df[['ds']])
        
        #Plot of the forecast with actual data
        graph.forecast_with_actual_data(model, test_df, forecast_on_test)
        
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
        graph.roc_cure(fpr_logistic, tpr_logistic,roc_auc_logistic, fpr_xgb, tpr_xgb, roc_auc_xgb)