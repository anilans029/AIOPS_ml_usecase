from src.utils.all_utils import read_yaml,save_data_local,create_directory
import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split

def spit_data(config_path,params_path):
    config = read_yaml(config_path)
    params = read_yaml(params_path)

    # get the directory_path from config file
    artifacts_dir = config["artifacts"]["artifacts_dir"]
    raw_local_dir = config["artifacts"]["raw_local_dir"]
    raw_local_fie = config["artifacts"]["raw_local_fie"]

    split_data_dir = config["artifacts"]["split_data_dir"]
    test_data_file = config["artifacts"]["test_data_file"]
    train_data_file = config["artifacts"]["train_data_file"]

    test_file_path = os.path.join(artifacts_dir, split_data_dir, test_data_file)
    train_file_path = os.path.join(artifacts_dir,split_data_dir,train_data_file)

    # get the params from params file
    test_size_param = params["base"]["test_size"]
    random_state_param = params["base"]["random_state"]

    create_directory([os.path.join(artifacts_dir, split_data_dir)])

    raw_local_file_path = os.path.join(artifacts_dir,raw_local_dir, raw_local_fie)
    df = pd.read_csv(raw_local_file_path,sep = ";")
    
    train, test =  train_test_split(df, test_size=test_size_param, random_state=random_state_param)

    for df, data_path in (train,train_file_path),(test,test_file_path):
        save_data_local(data=df,data_path=data_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c" , default="config/config.yaml")
    args.add_argument("--params", "-p" , default="params.yaml")
    parsed_args = args.parse_args()

    spit_data(parsed_args.config, parsed_args.params)