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
    
    houses = data['Hogwarts House'].unique()
    
    # Setup the figure and axes
    fig, axes = plt.subplots(4, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for i, course in enumerate(courses):
        ax = axes[i]
        for house in houses:
            subset = data[data['Hogwarts House'] == house]
            # Drop NaNs for the histogram
            ax.hist(subset[course].dropna(), alpha=0.5, label=house)
        
        ax.set_title(course)
        ax.legend()
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig("histograms.png", dpi=300)
    # plt.show()

if __name__ == "__main__":
    main()
