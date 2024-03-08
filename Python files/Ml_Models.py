import pandas as pd
import math as math
from sklearn.linear_model import LinearRegression, LogisticRegression #, SGDRegressor, Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from xgboost import XGBClassifier

class Models:

    def __init__(self) -> None:
        pass
    

    def logistic_regression(self, x_train : pd.DataFrame, x_test : pd.DataFrame, y_train : pd.Series, y_test : pd.Series):

        self.__shape_validation(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        print("Logistic Regression : ")
        logistic_reg = LogisticRegression()
        logistic_reg.fit(x_train, y_train)

        train_pred = logistic_reg.predict(x_train)
        test_pred = logistic_reg.predict(x_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        print(f'Accuracy of Training: {train_acc}')
        print(f'Accuracy of Testing: {test_acc}')
    

    def xgbclassifier(self, x_train : pd.DataFrame, x_test : pd.DataFrame, y_train : pd.Series, y_test : pd.Series):

        self.__shape_validation(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        print("xgbclassifier : ")
        xgbclassifier = XGBClassifier()
        xgbclassifier.fit(x_train, y_train)

        train_pred_xgb = xgbclassifier.predict(x_train)
        test_pred_xgb = xgbclassifier.predict(x_test)

        train_acc_xgb = accuracy_score(y_train, train_pred_xgb)
        test_acc_xgb = accuracy_score(y_test, test_pred_xgb)

        print("Evaluation results for XGBClassifier:")
        print(f"training set accuracy: {train_acc_xgb}")
        print(f"test set accuracy: {test_acc_xgb}")

    
    def linear_regression_ols(self, x_train : pd.DataFrame, x_test : pd.DataFrame, y_train : pd.Series, y_test : pd.Series):

        self.__shape_validation(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        print("Linear Regression OLS Model : ")
        ols = LinearRegression()
        ols.fit(x_train,y_train)
        print('RMSE in Train: '+str(round(math.sqrt(mean_squared_error(y_train,ols.predict(x_train))),2)))
        print('R2 in Train: '+str(round(r2_score(y_train,ols.predict(x_train)),2)))

        print('RMSE in Test: '+ str(round(math.sqrt(mean_squared_error(y_test,ols.predict(x_test))),2)))
        print('R2 in Test: '+ str(round(r2_score(y_test,ols.predict(x_test)),2)))
    

    def __shape_validation(self, x_train_shape : tuple, x_test_shape : tuple, y_train_shape : tuple, y_test_shape : tuple):
        if(x_train_shape[0] != y_train_shape[0]):
            raise Exception("Train Dataset Size Mismatch")
        if(x_test_shape[0] != y_test_shape[0]):
            raise Exception("Test Dataset Size Mismatch")
    
    def print(self):
        print("12345")
