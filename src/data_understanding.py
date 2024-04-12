import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import get_data, get_specific_data, display_kpis, setup_tabs
import warnings  # Adding warning ignore to avoid issues with distplot
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from graphs import Graphs
from models.ml_models import Models
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import ConvergenceWarning
from dashboard import Dashboard

warnings.filterwarnings('ignore')

# This will ignore all convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class DataUnderstanding:

    def __init__(self) -> None:
        # instantiating Graph & Model classes
        self.graph = Graphs()
        self.model = Models()
        self.dashboard = Dashboard()

    @staticmethod
    def load_and_filter_data(file_name: str):
        # Read dataset and display basic informations
        df = get_data(file_name)
        # Filtering data for only Litecoin
        df = get_specific_data(df, 'Litecoin')
        return df

    @staticmethod
    def clean_data(df: pd.DataFrame):
        # Removing columns we wont use because they have only null values
        df = df.drop(columns=["volume"])

        # Convert date object type to date type
        df['date'] = pd.to_datetime(df['date'])
        return df

    @staticmethod
    def handle_missing_data(df: pd.DataFrame):
        # Created a new dataframe to store newly create date column
        new_df = pd.DataFrame()
        new_df['m_date'] = df['date']

        # In this section we will check for the number of missing dates
        new_df = new_df.set_index('m_date')

        # Create a full date range from the start to the end of your dataset
        date_range = pd.date_range(start=new_df.index.min(), end=new_df.index.max())

        # Identify missing dates by finding those which are not in your Datafreme's index
        missing_dates = date_range.difference(new_df.index)
        print(f'Missing Date Size{missing_dates.size}')

        # On this other section we will use linear interpolation t add the missing dates and values for each column
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
        return df

    @staticmethod
    def create_features(df: pd.DataFrame):
        # Extract the year from the 'date' column using the dt accessor in pandas
        df['year'] = df['date'].dt.year
        # Extract the month from the 'date' column using the dt accessor in pandas
        df['month'] = df['date'].dt.month
        # Extract the day from the 'date' column using the dt accessor in pandas
        df['day'] = df['date'].dt.day
        # new features derived using existing features
        df['open_close'] = df['open'] - df['close']
        df['low_high'] = df['low'] - df['high']
        df['MA7'] = df['close'].rolling(window=7).mean()
        df['MA30'] = df['close'].rolling(window=30).mean()
        df['Price_Change'] = df['close'].diff()
        df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        df['PriceDifference'] = df['open'] - df['close']
        df['UpDown'] = np.where(df['PriceDifference'] > 0, 0, 1)
        return df

    def data_understanding(self, df: pd.DataFrame):
        # To decide if we want to show plots or not
        show_figure = False

        # Plot historical close price
        self.graph.basicPlot(y=df['close'], title='Crypto Close Price', x_label="years", y_label='Price in Dollars',
                             show_figure=show_figure)

        # Define features for future use
        features = ['open', 'high', 'low', 'close']
        self.graph.distPlotWithSubPlot(df, features=features, rows=2, cols=2, show_figure=show_figure)

        # Group the DataFrame 'df' by the 'year' column and calculate the mean of each numeric column for each group
        # numeric_only is to calculate mean only for numbers
        data_grouped = df.groupby(by=['year']).mean(numeric_only=True)

        bar_plot_features = ['open', 'high', 'low', 'close']
        # Plot a bar chart for the current column using the grouped data
        self.graph.barplotWithSubplot(data_grouped, bar_plot_features, 2, 2, show_figure=show_figure)

        # Keeping the columns for heatmap exploration
        sub_df = df[
            ['open', 'high', 'low', 'close', 'marketCap', 'MA7', 'MA30', 'Price_Change', 'low_high', 'year',
             'month', 'day', 'UpDown', ]]

        # Correlation for Litecoin crypto
        self.graph.graphCorrelation(sub_df.iloc[:, 1:], "Correlation HeatMap for Litecoin", show_figure=show_figure)

        visualize_cols = ['open', 'high', 'low', 'marketCap']

        # ploting graph to check correlation
        self.graph.scatterPlotWithSubPlot(sub_df, 'close', visualize_cols, 2, 2, show_figure=show_figure)

        # boxplot to check outliers with whisker_length(whis) of 1.5(default value)
        self.graph.boxPlotWithSubplot(sub_df, visualize_cols, 2, 2, show_figure=show_figure)

    def train_models(self, df: pd.DataFrame):
        model = self.model
        graph = self.graph
        show_figure = False
        sub_df = df[
            ['open', 'high', 'low', 'close', 'marketCap', 'year', 'month', 'day', 'UpDown']]

        # feature and target variables for classification
        x = df[['marketCap', 'open', 'year', 'month', 'day']]
        y = df['UpDown']

        scaler = StandardScaler()
        x = scaler.fit_transform(x)

        # calculate the index for the split
        splitting_index = int(len(df) * 0.8)

        # split data without shuffling
        x_train = x[:splitting_index]
        x_test = x[splitting_index:]
        y_train = y[:splitting_index]
        y_test = y[splitting_index:]

        y_train_class = y_train
        y_test_class = y_test

        # LogisticRegression
        logistic_reg = model.logistic_regression(x_train, x_test, y_train_class, y_test_class)

        train_acc_logistic, test_acc_logistic, y_pred_proba_logistic, trained_logistic_reg = logistic_reg
        print(f'Accuracy of Training: {train_acc_logistic}')
        print(f'Accuracy of Testing: {test_acc_logistic}')

        # use XGBClassifier to tain a model and predict classes
        reg_lambda = 1.0
        reg_alpha = 0.5
        learning_rate = 0.01
        max_depth = 3
        xgbclassifier = model.xgbclassifier(reg_lambda, reg_alpha, learning_rate, max_depth, x_train, x_test,
                                            y_train_class, y_test_class)
        train_acc_xgb, test_acc_xgb, y_pred_proba_xgb, trained_xgb_classifier = xgbclassifier
        print("Evaluation results for XGBClassifier:")
        print(f"training set accuracy: {train_acc_xgb}")
        print(f"test set accuracy: {test_acc_xgb}")

        # Random Forest Algorithem
        rf_train_acc, rf_test_acc, rf_y_pred, traind_classifer_rf = model.random_forest(x_train, x_test, y_train_class,
                                                                                        y_test_class)
        print(" Random Forest -- Train Accuracy ==>> ", rf_train_acc)
        print(" Random Forest -- Test Accuracy ==>> ", rf_test_acc)

        y_pred_knn, train_acc_knn, test_acc_knn, knn_model = model.knn(x_train, x_test, y_train_class, y_test_class)

        # Calculating ROC curve and AUC for Logistic
        fpr_logistic, tpr_logistic, thresholds_logistic = roc_curve(y_test_class, y_pred_proba_logistic)
        roc_auc_logistic = roc_auc_score(y_test_class, y_pred_proba_logistic)

        # Calculating ROC curve and AUC for XGBoost
        fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test_class, y_pred_proba_xgb)
        roc_auc_xgb = roc_auc_score(y_test_class, y_pred_proba_xgb)

        # Plotting ROC curve for both models
        graph.roc_cure(fpr_logistic, tpr_logistic, roc_auc_logistic, fpr_xgb, tpr_xgb, roc_auc_xgb,
                       show_figure=show_figure)

        # ploting Roc-curve for Random Forest
        fpr_rf, tpr_rf, threshold_rf = roc_curve(y_test_class, rf_y_pred)

        # Area under curve (auc)
        roc_auc_rf = roc_auc_score(y_test_class, rf_y_pred)
        graph.roc_cure_for_one_model(fpr_rf, tpr_rf, roc_auc_rf, "Random Forest")

        # Calculate ROC curve and AUC for KNN
        fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test_class, y_pred_knn)
        roc_auc_knn = roc_auc_score(y_test_class, y_pred_knn)

        # Plot ROC curve for KNN
        graph.roc_cure_for_one_model(fpr_knn, tpr_knn, roc_auc_knn, "KNN")
        logistic_fig, logistic_recall, logistic_precison = \
            model.display_classificiation_metrics(trained_logistic_reg, x_test, y_test_class,
                                                  model_name="Logistic Regression")

        xgbclassifier_fig, xgbclassifier_recall, xgbclassifier_precison = model.display_classificiation_metrics(
            trained_xgb_classifier, x_test, y_test_class, model_name="XGB Classifier")

        knn_fig, knn_recall, knn_precison = model.display_classificiation_metrics(knn_model, x_test, y_test_class,
                                                                                  model_name="knn")
        rf_fig, rf_recall, rf_precison = model.display_classificiation_metrics(traind_classifer_rf, x_test,
                                                                               y_test_class, "Random Forest")

        return ((train_acc_logistic, test_acc_logistic, logistic_recall, logistic_precison, logistic_fig, fpr_logistic,
                 tpr_logistic, roc_auc_logistic),
                (train_acc_xgb, test_acc_xgb, xgbclassifier_recall, xgbclassifier_precison, xgbclassifier_fig, fpr_xgb,
                 tpr_xgb, roc_auc_xgb),
                (rf_train_acc, rf_test_acc, rf_recall, rf_precison, rf_fig, fpr_rf, tpr_rf, roc_auc_rf),
                (train_acc_knn, test_acc_knn, knn_recall, knn_precison, knn_fig, fpr_knn, tpr_knn, roc_auc_knn),
                (df, sub_df))
