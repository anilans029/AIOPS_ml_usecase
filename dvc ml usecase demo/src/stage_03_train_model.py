from src.utils.all_utils import read_yaml,save_data_local,create_directory
import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import joblib

def train_model(config_path,params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    # get the directory_path from config file
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    split_data_dir = config["artifacts"]["split_data_dir"]
    train_data_file = config["artifacts"]["train_data_file"]
    trained_model_dir = config["artifacts"]["trained_model_dir"]
    trained_model_filename = config["artifacts"]["trained_model_filename"]

    train_file_path = os.path.join(artifacts_dir,split_data_dir,train_data_file)

    # get the params from params file
    random_state_param = params["base"]["random_state"]
    alpha_param = params["model_params"]["ElasticNet"]["alpha_param"]
    l1_ration_param = params["model_params"]["ElasticNet"]["l1_ration_param"]
    


    train = pd.read_csv(train_file_path, sep = ",")
    y_train = train["quality"]
    x_train = train.drop(columns = ["quality"])

    lr_model = ElasticNet(alpha=alpha_param, l1_ratio=l1_ration_param)
    lr_model.fit(x_train,y_train)



    model_dir_path = os.path.join(artifacts_dir, trained_model_dir) 
    create_directory([model_dir_path])
    model_file_path = os.path.join(model_dir_path, trained_model_filename)

    joblib.dump(lr_model, model_file_path)



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c" , default="config/config.yaml")
    args.add_argument("--params", "-p" , default="params.yaml")
    parsed_args = args.parse_args()

    train_model(parsed_args.config, parsed_args.params)