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
    
    # Filter data to remove NaNs in the courses columns
    data = data.dropna(subset=courses)
    
    houses = data['Hogwarts House'].unique()
    # Define colors for each house
    colors = {
        'Gryffindor': 'red',
        'Hufflepuff': 'yellow', 
        'Ravenclaw': 'blue', 
        'Slytherin': 'green'
    }
    
    n = len(courses)
    # Create a grid of subplots
    fig, axes = plt.subplots(n, n, figsize=(24, 24))
    
    print("Generating pair plot... this might take a moment.")
    
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            feature_x = courses[j]
            feature_y = courses[i]
            
            for house in houses:
                subset = data[data['Hogwarts House'] == house]
                # Use a default color if house not in dictionary
                c = colors.get(house, 'gray')
                
                if i == j:
                    # Diagonal: Histogram
                    ax.hist(subset[feature_x], alpha=0.5, color=c)
                else:
                    # Off-diagonal: Scatter plot
                    ax.scatter(subset[feature_x], subset[feature_y], s=1, alpha=0.5, color=c)
            
            # Formatting axes
            if i == n - 1:
                # Bottom row: show x labels
                # Shorten long names for display if needed, or rotate
                ax.set_xlabel(feature_x[:10], fontsize=8, rotation=45)
            else:
                ax.set_xticks([])
                
            if j == 0:
                # Left column: show y labels
                ax.set_ylabel(feature_y[:10], fontsize=8, rotation=45)
            else:
                ax.set_yticks([])
                
    plt.tight_layout()
    plt.savefig("pair_plot.png", dpi=300)
    # plt.show()

if __name__ == "__main__":
    main()
