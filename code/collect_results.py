import wandb
import numpy as np
import pandas as pd
import argparse
import time


def get_runs():
    """
    Collect Runs from WandB.

    Output:
        - runs from WandB.
    """
    api = wandb.Api()

    # Define the project
    project_name = "ThesisFinal2Grav+coll"
    project = f"nreints/{project_name}"

    runs = api.runs(project)
    print(f"There are {len(runs)} runs in {project_name}.")
    return runs


def get_grouped_filtered_runs(runs, filters, groups):
    """
    Group and filter runs.

    Input:
        - runs: runs from WandB.
        - filters: dictionary with filters.
        - groups: variables to group by.

    Output:
        - Filtered and grouped runs.
    """
    filtered_runs = []
    grouped_runs = {}
    for run in runs:
        config = run.config
        # If it satisfies all filters, save it
        if (
            all([config.get(filter) == filters[filter] for filter in filters])
            and ("lessSimulations??" not in run.tags)
            and (run.state != "crashed")
            and (run.state != "failed")
        ):
            filtered_runs += [run]
            config = run.config

            # Place run in correct group
            key = tuple([config.get(group) for group in groups])
            if key in grouped_runs:
                grouped_runs[key].append(run)
            else:
                grouped_runs[key] = [run]
    print(f"{len(filtered_runs)} runs satisfy the filters {filters}.")
    return filtered_runs, grouped_runs


def average_runs(group_dict, data_dir):
    """
    Calculate the average minimum of each group.

    Input:
        - group_dict: dictionary with grouped runs.
        - data_dir: directory of train data.
    """
    data = {
        "reference": ["" for _ in range(len(group_dict))],
        "str_extra_input": ["" for _ in range(len(group_dict))],
        "focus_identity": ["" for _ in range(len(group_dict))],
        "data_type": ["" for _ in range(len(group_dict))],
        "data_dir_train": ["" for _ in range(len(group_dict))],
        "number_of_runs": [0 for _ in range(len(group_dict))],
    }
    iter = 0
    for key, runs in (group_dict).items():
        data["data_dir_train"][iter] = key[0]
        data["reference"][iter] = key[1]
        data["str_extra_input"][iter] = key[2]
        data["focus_identity"][iter] = key[3]
        data["data_type"][iter] = key[4]
        print(f"Considering {len(runs)} runs of group {key}")

        all_losses = {
            key: np.zeros((len(runs), 1))
            for key in runs[0].history().keys()
            if "loss" in key
        }
        for i, run in enumerate(runs):
            history = run.history()
            for loss in all_losses.keys():
                # print(history.get(loss))
                # print(history.get(loss).dtype)
                if history.get(loss).dtype != "float64":
                    print(f"Could not do {loss} of {key} because of NaN")
                    continue
                all_losses[loss][i] = min(history.get(loss))
        data["number_of_runs"][iter] = i + 1
        for loss_name, min_vals in all_losses.items():
            # print(min_vals, loss_name)
            mean_min = np.mean(min_vals)
            mean_sqrt = np.sqrt(mean_min)
            # print("mean", mean_min)
            std_min = np.std(min_vals)
            # print("std ", std_min)
            if loss_name not in data.keys():
                data[loss_name] = ["" for _ in range(len(group_dict))]
            data[loss_name][iter] = "{:.4e} ".format(
                mean_min
            ) + "($\pm$ {:.4e})".format(std_min)

        iter += 1
    df = pd.DataFrame(data)
    df.to_pickle("results.pickle")
    return df


def get_specific_values(filter_dict, *rest):
    print(filter_dict)
    if len(rest) == 0:
        data = pd.read_pickle("results.pickle")
    else:
        data = rest[0]
    mask = np.logical_and.reduce(
        [
            pd.isnull(data[k]) if v is None else data[k] == v
            for k, v in filter_dict.items()
        ]
    )
    return data.loc[mask]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--new_collect_results",
        action="store_true",
        help="Force the model to focus on identity",
    )
    args = parser.parse_args()

    # Define train directory.
    # train_dir = "data_t(5,20)_r(0,0)_combi_pNone_gNone"
    # train_dir = "data_t(0,0)_r(5,20)_combi_pNone_gNone"
    # train_dir = "data_t(5,20)_r(5,20)_combi_pNone_gNone"
    # train_dir = "data_t(0,0)_r(0,0)_combi_pNone_gTrue"
    # train_dir = "data_t(5,20)_r(5,20)_combi_pTrue_gTrue"
    train_dir = "data_t(0,0)_r(5,20)_combi_pNone_gTrue"
    # train_dir = "data_tennis_pNone_gNone_tennisEffect"

    # Define filters
    filters = {
        "str_extra_input": False,
        # "focus_identity": True,
        # "focus_identity": False,
        "reference": "fr-fr",
        "data_dir_train": train_dir,
    }

    # Collect results from WandB
    if args.new_collect_results:
        start_time = time.time()
        # Collect runs
        runs = get_runs()
        group_by = [
            "data_dir_train",
            "reference",
            "str_extra_input",
            "focus_identity",
            "data_type",
        ]
        # Filter and group runs
        filtered_runs, grouped_runs = get_grouped_filtered_runs(
            runs,
            {
                "reference": "fr-fr",
                "str_extra_input": False,
                # "focus_identity": True,
                # "focus_identity": False,
                "data_dir_train": train_dir,
            },
            group_by,
        )
        # Calculate average minimum of each group
        average_data = average_runs(grouped_runs, train_dir[5:])

        # Filter extra if necessary.
        specific_df = get_specific_values(
            filters,
            average_data,
        )
        print(f"It took {time.time() - start_time} seconds to get all results!")
    # Calculate from already collected results
    else:
        # Filter runs.
        print("Using already collected runs !(These may be old)!")
        specific_df = get_specific_values(filters)

    # drop the columns where all values are NaN
    remove_empty_df = specific_df.replace("", np.nan)
    remove_empty_df = remove_empty_df.dropna(how="all", axis=1)
    print(remove_empty_df)
    # Store as CSV file
    remove_empty_df.to_csv("try_out.csv", index=False)
