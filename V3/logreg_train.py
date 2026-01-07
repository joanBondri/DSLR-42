#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse


class LogisticRegressionTrainer:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        
    def preprocess_data(self, df):
        """Preprocess the input DataFrame and return feature matrix X and label vector y"""

    def _normalize(self, X):
        """Normalize features to have mean 0 and variance 1"""
        
    def _sigmoid(self, z):
        """Compute the sigmoid function"""
        
    def cost_function(self, X, y):
        """Compute the cost for given X and y"""
        
    def train(self, X, y):
        """Train the logistic regression model using gradient descent"""
    
    def save_weights(self, file_path):
       """Save the learned weights to a file"""

def main():
    parser = argparse.ArgumentParser( formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('file', type=str, help='Path to csv file to train on')
    args = parser.parse_args()
    
    logisticRegressionTrainer = LogisticRegressionTrainer()
    df = pd.read_csv(args.file)
    

if __name__ == '__main__':
    main()