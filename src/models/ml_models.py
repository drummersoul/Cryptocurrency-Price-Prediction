import pandas as pd
import math as math
from sklearn.linear_model import LinearRegression, LogisticRegression #, SGDRegressor, Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    r2_score,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_fscore_support
)
from xgboost import XGBClassifier
from yellowbrick.classifier import ClassPredictionError
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

class Models:

    def __init__(self) -> None:
        pass
    

    def logistic_regression(self, x_train : pd.DataFrame, x_test : pd.DataFrame, y_train : pd.Series, y_test : pd.Series):

        self.__shape_validation(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        model_name = "Logistic Regression"
        print("Logistic Regression : ", end="\n\n")
        logistic_reg = LogisticRegression()
        logistic_reg.fit(x_train, y_train)

        train_pred = logistic_reg.predict(x_train)
        test_pred = logistic_reg.predict(x_test)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        y_pred_proba_logistic = logistic_reg.predict_proba(x_test)[:, 1]

        return train_acc, test_acc, y_pred_proba_logistic, logistic_reg

    def xgbclassifier(self, reg_lambda: float, reg_alpha: float, learning_rate: float, max_depth: int, x_train : pd.DataFrame, x_test : pd.DataFrame, y_train : pd.Series, y_test : pd.Series):

        self.__shape_validation(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        model_name = "XGB Classifier"
        print("xgbclassifier : ", end="\n\n")
        xgbclassifier = XGBClassifier(
                            reg_lambda = reg_lambda,
                            reg_alpha = reg_alpha,
                            learning_rate = learning_rate,
                            max_depth = max_depth
                        )
        xgbclassifier.fit(x_train, y_train)

        train_pred_xgb = xgbclassifier.predict(x_train)
        test_pred_xgb = xgbclassifier.predict(x_test)

        train_acc_xgb = accuracy_score(y_train, train_pred_xgb)
        test_acc_xgb = accuracy_score(y_test, test_pred_xgb)

        xgbclassifier_model = xgbclassifier
        y_pred_proba_xgb = xgbclassifier_model.predict_proba(x_test)[:, 1]

        self.class_prediction_error(x_train, x_test, x_test, y_test, xgbclassifier, 'XGBClassifier')
        return train_acc_xgb, test_acc_xgb, y_pred_proba_xgb, xgbclassifier

    
    def linear_regression_ols(self, x_train : pd.DataFrame, x_test : pd.DataFrame, y_train : pd.Series, y_test : pd.Series):

        self.__shape_validation(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        print("Linear Regression OLS Model : ", end="\n\n")
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

    def display_classificiation_metrics(self, trained_classifier, X_test, y_test, model_name=""):
        y_test_predicted = trained_classifier.predict(X_test)
        #display confusion matrix
        cm = confusion_matrix(y_test, y_test_predicted)
        # Create ConfusionMatrixDisplay object without plotting
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=trained_classifier.classes_)
        disp.plot()
        plt.title(f'Confusion Matrix for {model_name} Test Data')
        # Capture the current figure
        fig = plt.gcf()
        #print precision, recall, f1_score with respect to class with label 1 and accuracy
        precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_test_predicted, average='binary')
        print(f"classification metrics for {model_name} for test data:")
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"f1_score: {f1_score}\n")
        return fig, recall, precision

    def class_prediction_error(self, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame, model, model_name: str):
        visualizer = ClassPredictionError(model,  classes=['low', 'high'])

        # Fit the training data to the visualizer
        visualizer.fit(x_train, y_train)

        # Evaluate the model on the test data
        visualizer.score(x_test, y_test)

        # Draw visualization
        visualizer.show()


    def xgb_gcv(self, x_train : pd.DataFrame, x_test : pd.DataFrame, y_train : pd.Series, y_test : pd.Series):

        self.__shape_validation(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        param_grid = {
            'max_depth': [-1, 3, 5],
            'learning_rate': [0.001,0.01,0.05], 
            'n_estimators': [500,1000], #Number of fitted trees       
            'subsample': [0.8, 0.9], #Subsample number           
            'colsample_bytree': [0.8, 0.9] #number of colsample bytree
        }

        xgb_grid = XGBClassifier(random_state = 42)
        # Setup RandomizedSearchCV
        grid_search = GridSearchCV(
            estimator=xgb_grid, 
            scoring= 'f1',
            param_grid=param_grid, 
            cv=10, 
            verbose=1, 
            n_jobs=-1  # Use all available cores
        )

        self.display_grid_search_result(grid_search, x_train, x_test, y_train, y_test)

    def random_forest(self, x_train : pd.DataFrame, x_test : pd.DataFrame, y_train : pd.Series, y_test : pd.Series):

        self.__shape_validation(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        model_name = "Random Forest"
        print("Random Forest : ", end="\n\n")
        rf = RandomForestClassifier()
        rf.fit(x_train, y_train)

        rf_train_pred = rf.predict(x_train)
        rf_y_pred = rf.predict(x_test)

        train_acc = accuracy_score(y_train, rf_train_pred)
        test_acc = accuracy_score(y_test, rf_y_pred)

        return train_acc, test_acc, rf_y_pred, rf

    def logr_gcv(self, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):

        self.__shape_validation(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        # Define logistic regression model
        logistic_reg_model = LogisticRegression()

        # Define the search space for hyperparameters
        param_grid = {
            'penalty': ['l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1, 5, 10, 100],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [100, 200, 500],
        }

        # Define scoring metrics
        scoring = ['f1', 'neg_log_loss']  # Example metrics
        # Create GridSearchCV object
        grid_search = GridSearchCV(
                    estimator=logistic_reg_model,
                    param_grid=param_grid,
                    scoring=scoring,
                    refit='f1',  
                    cv=10,
                    verbose=0,
                    n_jobs=-1)
        
        self.display_grid_search_result(grid_search, x_train, x_test, y_train, y_test)
        

    def display_grid_search_result(self, grid_search, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):

        # Perform the random search
        grid_search.fit(x_train, y_train)

        # Best parameters and best score
        print("Best parameters found: ", grid_search.best_params_)
        print("Best score found: ", grid_search.best_score_)

        print('Accuracy of training is '+ str(accuracy_score(y_train,grid_search.predict(x_train))))
        print('Precision of training is '+ str(precision_score(y_train,grid_search.predict(x_train))))
        print('Recall of training is '+ str(recall_score(y_train,grid_search.predict(x_train))))
        print('F1 of training is '+ str(f1_score(y_train,grid_search.predict(x_train))))
        print('ROC AUC of training is '+ str(roc_auc_score(y_train,grid_search.predict(x_train))))
        print()
        print('Accuracy of test is '+ str(accuracy_score(y_test,grid_search.predict(x_test))))
        print('Precision of test is '+ str(precision_score(y_test,grid_search.predict(x_test))))
        print('Recall of test is '+ str(recall_score(y_test,grid_search.predict(x_test))))
        print('F1 of test is '+ str(f1_score(y_test,grid_search.predict(x_test))))
        print('ROC AUC of test is '+ str(roc_auc_score(y_test,grid_search.predict(x_test))))

    def knn(self, x_train : pd.DataFrame, x_test : pd.DataFrame, y_train : pd.Series, y_test : pd.Series):

        self.__shape_validation(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        model_name = "KNN Model"
        print("KNN Model : ", end="\n\n")
        # Instantiate KNN model
        knn = KNeighborsClassifier(n_neighbors=5)

        # Train the model
        knn.fit(x_train, y_train)

        # Predict classes for test data
        y_pred_knn = knn.predict(x_test)

        # Evaluate model performance
        train_acc_knn = knn.score(x_train, y_train)
        test_acc_knn = knn.score(x_test, y_test)

        return y_pred_knn, train_acc_knn, test_acc_knn, knn