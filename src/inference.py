import os
import sys
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,log_loss
from sklearn.datasets import load_digits
from utils import load_config,load_model

def main():
    digits=load_digits()
    X,y=digits.data,digits.target
    
    model=load_model('models/model_train.pkl')
    y_predicted=model.predict(X)

    y_predicted_prob=model.predict_proba(X)

    #Calculating the accuracy  
    accuracy=accuracy_score(y,y_predicted)

    #Calculating the loss 
    loss=log_loss(y,y_predicted_prob) 

    print("Inference Results:")
    print(f"Accuracy:{accuracy:.3f}")
    print(f"Loss:{loss:.3f}")
    print(f"Total number of predictions:{len(y_predicted)}")

    print("\nClassification Report:")
    print(classification_report(y,y_predicted))

    #Comparison between actual and predicted values
    print("\nExample comparison between actual and predicted classification")
    for i in range(10):
        print(f"Sample{i}:Actual Class:{y[i]} & Predicted Class:{y_predicted[i]}")

    #Saving predictions in a csv file
    os.makedirs('results',exist_ok=True)
    np.savetxt('results/predictions.csv',y_predicted,delimiter=',',fmt='%d')
    print("\nPredictions saved as results/predictions.csv ")

if __name__ == '__main__':
    main()

