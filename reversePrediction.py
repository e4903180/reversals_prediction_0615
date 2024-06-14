import json
from preprocessor.preprocessor_pytorch import Preprocessor
from model.model_pytorch import Model
from postprocessor.postprocessor import Postprocesser
from evaluator.evaluator_pytorch import Evaluator
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import os
import time
import pickle

class ReversePrediction():
    def set_seed(self, seed_value):
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        random.seed(seed_value)
        torch.manual_seed(seed_value)

    def run(self, params):
        self.set_seed(42)
        preprocessor = Preprocessor(params)
        X_train, y_train, X_val, y_val, X_test, y_test, test_dates, X_newest, x_newest_date, y_date, test_dataset = preprocessor.get_multiple_data()

        start_time = time.time()
        model_wrapper = Model(params=params)
        model, history, y_preds, online_history = \
            model_wrapper.run(X_train, y_train, X_test, y_test, X_val, y_val)
        end_time = time.time()
        execution_time = end_time - start_time
        print(model, file=open('log_model.txt', 'a'))

        y_preds = torch.tensor(y_preds, dtype=torch.float32)

        # y_pred_newest = model.forward(X_newest)
        # y_pred_newest = torch.tensor(y_pred_newest, dtype=torch.float32)

        evaluator = Evaluator(params)
        results = evaluator.get_results(y_train, y_val, y_test, y_preds, test_dataset,
                                        test_dates, history, online_history,
                                        show=False)
        results.update({'execution_time': execution_time})
        print(results, file=open('log.txt', 'a'))
        results.update({'using_data': params})
        results_json = json.dumps(results, indent=4)
        with open(params.get('summary_save_path'), 'w') as f:
            f.write(results_json)
            
        return results

if __name__ == '__main__':
    open('progress.txt', 'w').close()
    open('log.txt', 'w').close()
    root_paths = ['DNN_Projects_many_to_one5']
    for root_path in root_paths:
        for floder in tqdm.tqdm(os.listdir(root_path), file=open('progress.txt', 'a')):
            first_path = os.path.join(root_path, floder)
            for subfloder in tqdm.tqdm(os.listdir(first_path), file=open('progress.txt', 'a')):
                try:
                    second_path = os.path.join(first_path, subfloder)
                    print(second_path, file=open('progress.txt', 'a'))
                    params = json.load(open(os.path.join(second_path, 'parameters.json'), 'r'))
                    reversePrediction = ReversePrediction()
                    reversePrediction.set_seed(42)
                    results = reversePrediction.run(params)
                    # response
                    print('done', file=open('progress.txt', 'a'))
                except Exception as e:
                    print(e, file=open('progress.txt', 'a'))
                    continue
                
    # root_path = 'DNN_Projects_models_STOXX'
    # for floder in tqdm.tqdm(os.listdir(root_path), file=open('progress.txt', 'a')):
    #     first_path = os.path.join(root_path, floder)
    #     for subfloder in tqdm.tqdm(os.listdir(first_path), file=open('progress.txt', 'a')):
    #         try:
    #             second_path = os.path.join(first_path, subfloder)
    #             print(second_path, file=open('progress.txt', 'a'))
    #             params = json.load(open(os.path.join(second_path, 'parameters.json'), 'r'))
    #             reversePrediction = ReversePrediction()
    #             reversePrediction.set_seed(42)
    #             results = reversePrediction.run(params)
    #             # response
    #             print('done', file=open('progress.txt', 'a'))
    #         except Exception as e:
    #             print(e, file=open('progress.txt', 'a'))
    #             continue
    

