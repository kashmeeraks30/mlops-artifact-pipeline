# Phase 1 - Training Pipeline

import json
import pickle
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
from utils import load_config,train_model

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