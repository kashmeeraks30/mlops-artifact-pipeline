import json
import pickle
import os
import sys
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from src.utils import load_config,train_model


def test_config_file_loading():
    #Test if the config file is loaded successfully
    config = load_config()
    assert config is not None
    assert isinstance(config,dict)

    #Test if all the required parameters exist in the config
    required_parameters = ['C','solver','max_iter']

    for param in required_parameters:
        assert param in config, f"Missing required parameter:{param}"

    #Test if the parameters are of correct datatypes
    assert isinstance(config['C'],(int,float))
    assert isinstance(config['solver'],str)
    assert isinstance(config['max_iter'],int)

def test_model_creation():
    
    digits=load_digits()
    X,y=digits.data,digits.target

    config=load_config()
    model=train_model(X,y,config)

    #Test if the train_model function returns a LogisticRegression object
    assert isinstance(model,LogisticRegression),"Model should be LogisticRegression"

    #Test if the model attributes are present
    assert hasattr(model,'coef_'),"Missing coef_ attribute, object must be fitted"
    assert hasattr(model,'classes_'), "Missing classes_ attribute, object must be fitted"

def test_accuracy():
    digits=load_digits()
    X,y=digits.data,digits.target
    
    config=load_config()
    model=train_model(X,y,config)

    y_predicted=model.predict(X)
    accuracy=accuracy_score(y,y_predicted)

    #Test if the accuracy is above a certain threshold which is 0.9
    assert accuracy>0.9, f"Model accuracy is below the set threshold"




