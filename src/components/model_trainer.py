import os
import sys
from dataclasses import dataclass

#Various Regression Algos.
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

#metrics
from sklearn.metrics import r2_score

#housekeeping
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation

#custom functions
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact','model.pkl')


class modelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split training and test input data')
            x_train,y_train, x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" :  DecisionTreeRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Nearest Neighbor Regressor" : KNeighborsRegressor(),
                "XGBClassifier" : XGBRegressor(),
                "CatBoosting" : CatBoostRegressor(verbose=False),
                "AdaBoost" : AdaBoostRegressor()
            }
            
            model_report:dict=evaluate_model(x_train,y_train,x_test, y_test,models)
            #Best Model Name
            best_model_score = max(sorted(model_report.values()))
            best_model_name = max(model_report, key= lambda i: model_report[i])
            best_model = models[best_model_name]
            #Passing threshold
            if best_model_score < 0.6:
                raise CustomException("No Model Qualifies")
            logging.info(f"best found model is {best_model} on both training and testing datasets")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            predicted=best_model.predict(x_test)
            r2_sq=r2_score(y_test, predicted)
            return r2_sq

        except Exception as e:
            raise CustomException(e, sys)