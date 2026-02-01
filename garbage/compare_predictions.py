#!/usr/bin/env python3
import pandas as pd
import argparse


def compare_predictions(predictions_file, truth_file, output_file):
    """
    Compare predicted houses with true houses and generate a comparison CSV

    Args:
        predictions_file: Path to predictions CSV (houses.csv)
        truth_file: Path to test dataset with true houses
        output_file: Path to output comparison CSV
    """
    # Load predictions
    predictions_df = pd.read_csv(predictions_file)

    # Load true values
    truth_df = pd.read_csv(truth_file)

    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Index': predictions_df['Index'],
        'Predicted_House': predictions_df['Hogwarts House'],
        'True_House': truth_df['Hogwarts House'],
    })

    # Add match column
    comparison_df['Match'] = comparison_df['Predicted_House'] == comparison_df['True_House']

    # Save to CSV
    comparison_df.to_csv(output_file, index=False)

    # Calculate and display accuracy
    total = len(comparison_df)
    matches = comparison_df['Match'].sum()
    accuracy = (matches / total) * 100

    print(f"Total predictions: {total}")
    print(f"Correct predictions: {matches}")
    print(f"Incorrect predictions: {total - matches}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"\nComparison saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare predicted houses with true houses'
    )
    parser.add_argument(
        '--predictions',
        type=str,
        default='houses.csv',
        help='Path to predictions CSV (default: houses.csv)'
    )
    parser.add_argument(
        '--truth',
        type=str,
        default='datasets/dataset_test.csv',
        help='Path to test dataset with true houses (default: datasets/dataset_test.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='garbage/comparison.csv',
        help='Path to output comparison CSV (default: garbage/comparison.csv)'
    )

    args = parser.parse_args()

    compare_predictions(args.predictions, args.truth, args.output)


if __name__ == '__main__':
    main()
