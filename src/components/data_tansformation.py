import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transfrmation_config= DataTransformationConfig()

    def get_transformer_object(self):
        try:
            categorical_columns= ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            numerical_columns= ['reading_score', 'writing_score']
            logging.info("pipeline for numerical columns")
            num_pipeline=Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="median")),
                    ("Scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info("pipeline for categorical columns")
            cat_pipeline=Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="most_frequent")),
                    ("Encoder",OneHotEncoder()),
                    ("Scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")

            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Pre-processing initiated, seperating target feature")

            pre_processing_obj=self.get_transformer_object()
            target_column="math_score"
            num_column=['reading_score','writing_score']

            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]

            logging.info("Applying feature transformation on training and testing data")

            input_feature_train_arr=pre_processing_obj.fit_transform(input_feature_train_df)

            input_feature_test_arr=pre_processing_obj.transform(input_feature_test_df)

            logging.info("Combining pre processed array and target")

            train_arr= np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr= np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info("saved pre-processing object")

            save_object(
                file_path=self.data_transfrmation_config.preprocessor_obj_file_path,obj=pre_processing_obj
            )

            logging.info("pre-processing completed")

            return(train_arr,test_arr,self.data_transfrmation_config.preprocessor_obj_file_path)
        
        except Exception as e:
            raise CustomException(e,sys)

