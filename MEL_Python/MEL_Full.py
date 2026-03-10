import numpy as np
import pandas as pd

class jMultiTaskPSO_Full:
    def __init__(self):
        pass

    # Add methods and attributes to implement the DSD-MOEA algorithm


def load_dataset_from_columns(path):
    # Load datasets from the specified path, treating columns as samples
    data = pd.read_csv(path)
    labels = []
    datasets = []

    for column in data.columns:
        if column.startswith('Tumor'):
            label = 'Tumor'
        elif column.startswith('Normal'):
            label = 'Normal'
        elif column.startswith('ALL'):
            label = 'ALL'
        elif column.startswith('AML'):
            label = 'AML'
        elif column.startswith('non-malignant'):
            label = 'non-malignant'
        elif column.startswith('tumor'):
            label = 'tumor'
        elif column.startswith('CTRL'):
            label = 'CTRL'
        elif column.startswith('IS'):
            label = 'IS'
        else:
            label = 'Unknown'

        labels.append(label)
        datasets.append(data[column].values)

    return np.array(datasets), np.array(labels)


if __name__ == '__main__':
    # Example of loading dataset and running the algorithm
    datasets, labels = load_dataset_from_columns('/kaggle/input/datasets/eminz132/dataset/Datasets_part1')

    for idx, dataset in enumerate(datasets):
        np.save(f'curve_full_dataset{idx}.npy', dataset) # Saving dataset
        np.save(f'hamming_curve_full_dataset{idx}.npy', dataset) # Saving hamming dataset
