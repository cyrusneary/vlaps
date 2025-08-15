import pathlib

import numpy as np

import matplotlib.pyplot as plt

import pickle

def read_libero_results(file_path: str) -> tuple:
    """
    Read the Libero results from a .txt file and return the data as a tuple of numpy arrays.
    The first column is the time, and the second column is the voltage.

    Args:
        file_path (str): Path to the .txt file containing the Libero results.

    Returns:
        tuple: A tuple containing two numpy arrays: time and voltage.
    """

    # Open the pickle file
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data

if __name__ == "__main__":
    experiments_file_dir = "experiments/libero/results"
    exp_name = "2025-04-27_21-03-59_libero_pi0fast"
    episode_name = "libero_90_36_failure"
    file_name = "all_sampled_logprobs.pkl"
    file_path = pathlib.Path(experiments_file_dir) / exp_name / episode_name / file_name
    print(exp_name)

    data = read_libero_results(file_path)
    print(data)