import argparse
import os
import pickle
import shutil


def get_nr_sims(dirs):
    """
    Returns the number of simulations to use per directory.

    Input:
        - dirs: list of data directories that need to be combined.

    Output:
        - new_nr_sims: number of simulations to use per directory.
    """
    nr_sims = []
    for dir in dirs:
        nr_sims.append(len(os.listdir("data/" + dir)))
    print(nr_sims)
    if nr_sims.count(nr_sims[0]) != len(nr_sims):
        new_nr_sims = min(nr_sims) // len(nr_sims)
        print(
            f"Not all directories contain the same number of simulations\n using {new_nr_sims} simulations from each directory."  # TODO
        )
    else:
        new_nr_sims = nr_sims[0] // len(nr_sims)
        print(f"Using {new_nr_sims} simulations from each directory.")

    return new_nr_sims


def create_combi(dir_list, new_dir):
    """
    Creates a combined dataset in new_dir.

    Input:
        - dir_list: list of data directories that need to be combined.
        - new_dir: name of the new directory.
    """
    assert len(set(dir_list)) == len(
        dir_list
    ), "At least 1 directory occurs more than once."
    print(f"Using {len(dir_list)} directories.")
    nr_sims = get_nr_sims(dir_list)
    shutil.rmtree(f"data/{new_dir}")
    os.makedirs(f"data/{new_dir}", exist_ok=True)

    new_id = 0
    for dir in dir_list:
        for id in range(nr_sims):
            with open(f"data/{dir}/sim_{id}.pickle", "rb") as f:
                sim_data = pickle.load(f)
                with open(f"data/{new_dir}/sim_{new_id}.pickle", "wb") as new_f:
                    pickle.dump(sim_data, new_f)
            new_id += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dirs", type=str)
    parser.add_argument("-n", "--new_dir", type=str)

    args = parser.parse_args()

    # new_data_dir = " ".join(args.new_dir)
    print("good?", args.dirs.split(";"))
    print(args.dirs)

    print(args.new_dir)
    # print(new_data_dir)
    # create_combi(args.dirs, new_data_dir)

    # print(f"-- Finished combining the datasets. Saved in {new_data_dir} --")
