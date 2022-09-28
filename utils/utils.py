from typing import List
import tensorflow as tf
import mlflow
from omegaconf import DictConfig, ListConfig
import pickle

def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)

def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)

def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)


if __name__ == "__main__":
    import numpy as np
    a = np.concatenate(
        [
            np.random.randint(0, 2, 10).reshape(-1, 1),
            np.random.randint(0, 7, 10).reshape(-1, 1),
            np.random.randint(0, 5, 10).reshape(-1, 1),
            np.random.randn(10).reshape(-1, 1),
            np.random.randn(10).reshape(-1, 1),
        ],
        axis=1
    )
    inputs = np.expand_dims(a, 0)
    tf.transpose(inputs)
    inputs.T[0]
    inputs[:, 0]

