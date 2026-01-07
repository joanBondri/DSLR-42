import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'V1'))
from describe import get_dataframe

def main():
    data = get_dataframe('../datasets/dataset_train.csv')

    courses = [
        'Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
        'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
        'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying'
    ]
    
    # Calculate correlation matrix
    corr_matrix = data[courses].corr().abs()
    
    # Find the pair with the highest correlation (excluding diagonal)
    max_corr = 0
    feature1 = ""
    feature2 = ""
    
    for i in range(len(courses)):
        for j in range(i + 1, len(courses)):
            corr = corr_matrix.iloc[i, j]
            if corr > max_corr:
                max_corr = corr
                feature1 = courses[i]
                feature2 = courses[j]
    
    print(f"Most similar features: {feature1} and {feature2} with correlation {max_corr}")
    
    # Plot scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(data[feature1], data[feature2], alpha=0.5)
    plt.title(f'Scatter Plot: {feature1} vs {feature2}')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.grid(True)
    # plt.show()
    plt.savefig("scatter.png", dpi=300)

if __name__ == "__main__":
    main()
