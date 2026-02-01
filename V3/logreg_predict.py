
import argparse
import pandas as pd
import numpy as np
import pickle

class LogisticRegressionPredictor:
    def __init__(self, weightsFile='weights.pkl', ):
        self.weightsFile = weightsFile
        
    def treatWeightsFile(self):
        with open(self.weightsFile, 'rb') as f:
            weightDatas = pickle.load(f)
        self.impute_means = weightDatas['impute_means']
        self.feature_columns = weightDatas['feature_columns']
        self.mean = weightDatas['mean']
        self.std = weightDatas['std']
        self.models = weightDatas['models']
            
    def preprocess_data(self, df):
        df = self._impute_mean(df)
        X = df[self.feature_columns].values
        X = self._normalize(X)
        return X

    def _impute_mean(self, df):
        df_copy = df.copy()
        for col in self.feature_columns:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].fillna(self.impute_means[col])
        return df_copy

    def _normalize(self, X):
        X_normalized = (X - self.mean) / self.std
        return X_normalized
        
    def _sigmoid(self, z):
        """Compute the sigmoid function"""
        return 1 / (1 + np.exp(-z))
    
    def predict(self, X):
        n_samples = X.shape[0]
        house_names = list(self.models.keys())
        n_houses = len(house_names)
        probabilities = np.zeros((n_samples, n_houses))
        for i, house in enumerate(house_names):
            model = self.models[house]
            z = np.dot(X, model['weights']) + model['bias']
            probabilities[:, i] = self._sigmoid(z)
        predicted_indices = np.argmax(probabilities, axis=1)
        predictions = [house_names[idx] for idx in predicted_indices]

        return predictions

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('weightsFile', type=str, help='Path to weight file to predict with')
    parser.add_argument('file', type=str, help='Path to csv file to predict with')
    args = parser.parse_args()
    df = pd.read_csv(args.file)
    predictor = LogisticRegressionPredictor(args.weightsFile)
    predictor.treatWeightsFile()
    X = predictor.preprocess_data(df)
    predictions = predictor.predict(X)
    output_df = pd.DataFrame({
        'Index': range(len(predictions)),
        'Hogwarts House': predictions
    })

    output_df.to_csv('houses.csv', index=False)
    print(f"Predictions saved to houses.csv ({len(predictions)} students)")

if __name__ == '__main__':
    main()