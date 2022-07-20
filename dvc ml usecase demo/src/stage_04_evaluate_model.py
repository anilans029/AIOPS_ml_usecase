from src.utils.all_utils import read_yaml,save_data_local,create_directory,save_report
import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
import joblib
import numpy as np

def evaluate_metrics_score(actual_values,Ypred):
    mse = mean_squared_error(actual_values, Ypred)
    rmse = np.sqrt(mse)
    r2Score = r2_score(actual_values, Ypred)
    mae = mean_absolute_error(actual_values,Ypred)
    return rmse, mse, r2Score,mae

def evaluate_model(config_path,params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    # get the directory_path from config file
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    split_data_dir = config["artifacts"]["split_data_dir"]
    test_data_file = config["artifacts"]["test_data_file"]
    trained_model_dir = config["artifacts"]["trained_model_dir"]
    trained_model_filename = config["artifacts"]["trained_model_filename"]
    report_dir = config["artifacts"]["report_dir"]
    report_file = config["artifacts"]["report_file"]

    test_file_path = os.path.join(artifacts_dir,split_data_dir,test_data_file)

    # get the params from params file
    alpha_param = params["model_params"]["ElasticNet"]["alpha_param"]
    l1_ration_param = params["model_params"]["ElasticNet"]["l1_ration_param"]
    


    test = pd.read_csv(test_file_path, sep = ",")
    y_test = test["quality"]
    x_test = test.drop(columns = ["quality"])

    model_dir_path = os.path.join(artifacts_dir, trained_model_dir) 
    model_file_path = os.path.join(model_dir_path, trained_model_filename)
    
    lr_model = joblib.load(model_file_path)
    y_pred = lr_model.predict(x_test)

    rmse, mse, r2Score,mae = evaluate_metrics_score(y_test, y_pred)

    report = {
        "rmse":rmse,
        "mse":mse,
        "r2Score":r2Score,
        "mae":mae
    }

    report_dir_path = os.path.join(artifacts_dir, report_dir)
    create_directory([report_dir_path])

    report_file_path = os.path.join(report_dir_path,report_file)
    save_report(report, report_file_path)




if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c" , default="config/config.yaml")
    args.add_argument("--params", "-p" , default="params.yaml")
    parsed_args = args.parse_args()

    evaluate_model(parsed_args.config, parsed_args.params)