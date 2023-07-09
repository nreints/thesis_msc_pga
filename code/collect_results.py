import wandb
import numpy as np
import pandas as pd
import argparse
import time


def get_runs():
    api = wandb.Api()

    # Define your project
    project_name = "ThesisFinal2"
    project = f"nreints/{project_name}"

    runs = api.runs(project)
    print(f"There are {len(runs)} runs in {project_name}.")
    return runs


def get_grouped_filtered_runs(runs, filters, groups):
    filtered_runs = []
    grouped_runs = {}
    for run in runs:
        config = run.config
        # print(("lessSimulations??" not in run.tags))
        # If it satisfies all filters, save it
        if all([config.get(filter) == filters[filter] for filter in filters]) and (
            "lessSimulations??" not in run.tags
        ):
            filtered_runs += [run]
            config = run.config

            key = tuple([config.get(group) for group in groups])

            if key in grouped_runs:
                grouped_runs[key].append(run)
            else:
                grouped_runs[key] = [run]
    print(f"{len(filtered_runs)} runs satisfy the filters {filters}.")
    return filtered_runs, grouped_runs


# def group_runs(runs, groups):
#     grouped_runs = {}
#     for run in runs:
#         config = run.config

#         key = tuple([config.get(group) for group in groups])

#         if key in grouped_runs:
#             grouped_runs[key].append(run)
#         else:
#             grouped_runs[key] = [run]
#     return grouped_runs


# def get_grouped_runs_average(runs):


def average_runs(group_dict, data_dir):
    # print(group_dict.keys())
    # exit()
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

        # for loss_type in key for key,val in run.history.items() if "loss" in key]
        # train_loss_values = np.zeros((len(runs), runs[0].config.get("epochs")))
        # test_loss_values = np.zeros((len(runs), runs[0].config.get("epochs")))
        all_losses = {
            key: np.zeros((len(runs), 1))
            for key in runs[0].history().keys()
            if "loss" in key
        }
        for i, run in enumerate(runs):
            history = run.history()
            for loss in all_losses.keys():
                if history.get(loss).dtype != "float64":
                    print(f"Could not do {loss} of {key} because of NaN")
                    continue
                all_losses[loss][i] = min(history.get(loss))
        data["number_of_runs"][iter] = i + 1
        for loss_name, min_vals in all_losses.items():
            mean_min = np.mean(min_vals)
            std_min = np.std(min_vals)
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
    # print(data)
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

    # train_dir = "data_t(5,20)_r(0,0)_combi_pNone_gNone"
    # train_dir = "data_t(0,0)_r(5,20)_combi_pNone_gNone"
    # train_dir = "data_t(5,20)_r(5,20)_combi_pNone_gNone"
    train_dir = "data_t(0,0)_r(0,0)_combi_pNone_gTrue"
    filters = {
        # "str_extra_input": False,
        # "focus_identity": True,
        # "focus_identity": False,
        # "reference": "start-fr",
        "data_dir_train": train_dir,
    }

    if args.new_collect_results:
        start_time = time.time()
        runs = get_runs()
        group_by = [
            "data_dir_train",
            "reference",
            "str_extra_input",
            "focus_identity",
            "data_type",
        ]
        filtered_runs, grouped_runs = get_grouped_filtered_runs(
            runs,
            {
                #     # "reference": "fr-fr",
                #     # "str_extra_input": None,
                #     # "focus_identity": False,
                #     "data_dir_train": train_dir
            },
            group_by,
        )
        average_data = average_runs(grouped_runs, train_dir[5:])

        specific_df = get_specific_values(
            filters,
            average_data,
        )
        print(f"It took {time.time() - start_time} seconds to get all results!")
    else:
        print("Using already collected runs !(These may be old)!")
        specific_df = get_specific_values(filters)

    print(specific_df)
    remove_empty_df = specific_df.replace("", np.nan)

    # drop the columns where all values are NaN
    remove_empty_df = remove_empty_df.dropna(how="all", axis=1)
    print(remove_empty_df)
    remove_empty_df.to_csv("try_out.csv", index=False)
    # history = run.history()
    #     print(history)
    #     for col in history:
    #         print(col)
    #     print(config)
    #     print(config.get("data_type"))
    #     print(history["Train loss"])
    #     exit()
