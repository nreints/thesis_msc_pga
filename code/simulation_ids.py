import os
import pickle
import argparse
import random
from utils import check_number_sims


def divide_train_test_sims(data_dir_train, data_dirs_test):
    """
    Divides the train and test simulation IDs.

    Input:
        - data_dir_train: data directory for the train data.
        - data_dirs_test: data directories for the test data.

    Output:
        - n_sims_train_total: total number of simulations in data_dir_train.
        - train_sims: list of IDs of the simulations used for training the model.
        - test_sims: list of IDs of the simulations used for testing the model.
    """
    n_sims_train_total = len(os.listdir(data_dir_train))
    print("Total number of simulations in train dir: ", n_sims_train_total)
    sims_train = range(0, n_sims_train_total)
    train_sims = random.sample(sims_train, int(0.8 * n_sims_train_total))
    test_sims = list(set(sims_train) - set(train_sims))
    check_number_sims(data_dir_train, train_sims, data_dirs_test, test_sims)
    print("Number of train simulations: ", len(train_sims))
    print("Number of test simulations: ", len(test_sims))
    return n_sims_train_total, train_sims, test_sims


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-train_dir",
        type=str,
        help="Train Directory",
        default="data/data_t(5,20)_r(5,20)_full_pNone_gNone",
    )
    parser.add_argument(
        "-test_dir",
        type=str,
        help="Test Directory",
        default="data_t(5,20)_r(5,20)_full_pNone_gNone",
    )
    parser.add_argument(
        "-file_name",
        type=str,
        help="File name where to store the simulation IDs",
        default="train_test_ids",
    )
    parser.add_argument("-iter", type=int, help="number of iterations")

    args = parser.parse_args()
    # os.makedirs(f"data/{args.file_name}", exist_ok=True)

    final_dict = {}
    for i in range(args.iter):
        # Get simulation ids
        n_sims_total, train_sim_IDs, test_sim_IDs = divide_train_test_sims(
            args.train_dir, [args.test_dir]
        )
        # Put in dictionary
        final_dict[i] = {"train_sims": train_sim_IDs, "test_sims": test_sim_IDs}

    # Save in pickle
    file_name = f"{args.file_name}_{n_sims_total}.pickle"
    if os.path.exists(file_name):
        os.remove(file_name)
    with open(file_name, "wb") as f:
        pickle.dump(final_dict, f)

    print(f"Simulation IDs saved in {file_name}")
