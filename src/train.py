# Phase 1 - Training Pipeline

import json
import pickle
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

def load_config(config_path='config/config.json'):
    with open(config_path,'r') as f:
        config=json.load(f)
    return config

def train_model(X,y,config):
    model=LogisticRegression(C=config['C'],solver=config['solver'],max_iter=config['max_iter'],random_state=42)
    model.fit(X,y)
    return model

def main():
    digits = load_digits()
    X,y = digits.data, digits.target
    
    #Loading configuration from JSON file
    config = load_config()
    
    #Training the model 
    model = train_model(X,y,config)

    #Evaluating the model
    y_predicted=model.predict(X)

    #Calculating the accuracy
    accuracy = accuracy_score(y,y_predicted)

    print(f'Training Accuracy: {accuracy:.4f}')

    os.makedirs('models',exist_ok=True)
    with open('models/model_train.pkl','wb') as f:
        pickle.dump(model,f)
    
    print("Model saved in models/model_train.pkl")
if __name__ == "__main__":
    main()