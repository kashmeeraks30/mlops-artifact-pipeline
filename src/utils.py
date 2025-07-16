import sys
import os
import json
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_config(config_path='config/config.json'):
    with open(config_path,'r') as f:
        config=json.load(f)
    return config

def train_model(X,y,config):
    model=LogisticRegression(C=config['C'],solver=config['solver'],max_iter=config['max_iter'],random_state=42)
    model.fit(X,y)
    return model