#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
import pickle


class LogisticRegressionTrainer:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.impute_means = {}  # Store means for imputation by column name

    def preprocess_data(self, df):
        df = self._impute_mean(df)

        # Select numeric columns excluding target column
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                       if col != 'Hogwarts House']
        self.feature_columns = feature_cols  # Save for later use in prediction
        X = df[feature_cols].values  # Convert to NumPy array

        y = df['Hogwarts House'].values

        X = self._normalize(X)

        return X, y

    def _impute_mean(self, df):
        """
        Replace the missing values on the dataset with the average value of that subject (numeric columns only)
        """
        df_copy = df.copy()

        # Only impute mean for numeric columns (courses)
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_mean = df_copy[col].mean()
            self.impute_means[col] = col_mean  # Save for later use in prediction
            df_copy[col] = df_copy[col].fillna(col_mean)

        return df_copy

    def _normalize(self, X):
        """
        Normalize features to have mean 0 and variance 1 (standardization)
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            X_normalized: Normalized feature matrix
        """
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
        # Avoid division by zero - if std is 0, keep the feature as is
        self.std = np.where(self.std == 0, 1, self.std)
        
        X_normalized = (X - self.mean) / self.std
        
        return X_normalized
        
    def _sigmoid(self, z):
        """Compute the sigmoid function"""
        return 1 / (1 + np.exp(-z))
        
    def cost_function(self, X, y):
        """Compute the cost for given X and y"""
        m = len(y)
        h = self._sigmoid(np.dot(X, self.weights) + self.bias)
        
        # Binary cross-entropy loss
        cost = (-1/m) * np.sum(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))
        return cost
        
    def train(self, X, y):
        """Train the logistic regression model using gradient descent"""
        m, n = X.shape
        
        self.weights = np.zeros(n)
        self.bias = 0
        
        for i in range(self.num_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(z)
            
            # Compute gradients
            dw = (1/m) * np.dot(X.T, (predictions - y))
            db = (1/m) * np.sum(predictions - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Printing cost every 100 iterations to debug
            if i % 100 == 0:
                cost = self.cost_function(X, y)
                print(f"Iteration {i}: Cost = {cost}")
    
    def save_weights(self, models, file_path):
        """Save the learned weights to a file"""

        weights_data = {
            'impute_means': self.impute_means,  # Means for imputation (dict)
            'feature_columns': self.feature_columns,  # Feature column names (list)
            'mean': self.mean,  # Means for normalization (numpy array)
            'std': self.std,  # Stds for normalization (numpy array)
            'models': {house: {'weights': models[house].weights, 'bias': models[house].bias}
                    for house in models}
        }

        with open(file_path, 'wb') as f:
            pickle.dump(weights_data, f)

        print("All models saved to weights.pkl")

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('file', type=str, help='Path to csv file to train on')
    args = parser.parse_args()
    
    df = pd.read_csv(args.file)
    
    logisticRegressionTrainer = LogisticRegressionTrainer(learning_rate=1, num_iterations=10000)
    X, y = logisticRegressionTrainer.preprocess_data(df)
    
    houses = np.unique(y)
    models = {}
    
    # Train one model per house (One-vs-All)
    for house in houses:
        print(f"\nTraining model for {house}...")
        # Create binary labels: 1 if this house, 0 otherwise
        y_binary = (y == house).astype(int)
        
        # Train model
        trainer = LogisticRegressionTrainer(learning_rate=1, num_iterations=10000)
        trainer.mean = logisticRegressionTrainer.mean
        trainer.std = logisticRegressionTrainer.std
        trainer.train(X, y_binary)
        
        models[house] = trainer
    
    logisticRegressionTrainer.save_weights(models, 'weights.pkl')
    

if __name__ == '__main__':
    main()