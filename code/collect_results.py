import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse


def get_runs():
    api = wandb.Api()

    # Define your project
    project = "nreints/ThesisFinal"

    runs = api.runs(project)
    print(f"There are {len(runs)} runs in ThesisFinal.")
    return runs


def get_grouped_filtered_runs(runs, filters, groups):
    filtered_runs = []
    grouped_runs = {}
    for run in runs:
        config = run.config

        # If it satisfies all filters, save it
        if all([config.get(filter) == filters[filter] for filter in filters]):
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


def average_runs(group_dict):
    data = {
        "reference": [],
        "str_extra_input": [],
        "identity_focus": [],
        "data_type": [],
        "data_dir_train": [],
        "mean_min_train": [],
        "mean_min_test": [],
    }
    for key, runs in (group_dict).items():
        data["reference"] += [key[0]]
        data["str_extra_input"] += [key[1]]
        data["identity_focus"] += [key[2]]
        data["data_type"] += [key[3]]
        print(f"Considering runs of group {key}")
        train_loss_values = np.zeros((len(runs), runs[0].config.get("epochs")))
        test_loss_values = np.zeros((len(runs), runs[0].config.get("epochs")))
        for i, run in enumerate(runs):
            history = run.history()
            train_loss_values[i] = history.get("Train loss")
            test_loss_values[i] = history.get(
                "Test loss t(5,20)_r(5,20)_combi_pNone_gNone"
            )
        data["data_dir_train"] = run.config["data_dir_train"]

        min_vals_train = np.min(train_loss_values, axis=1)
        mean_min_train = np.mean(min_vals_train)
        data["mean_min_train"] = mean_min_train
        min_vals_test = np.min(test_loss_values, axis=1)
        mean_min_test = np.mean(min_vals_test)
        data["mean_min_test"] = mean_min_test

    df = pd.DataFrame(data)
    df.to_pickle("results.pickle")
    return df

    #     mean_group_train = np.mean(train_loss_values, axis=0)
    #     std_group_train = np.std(train_loss_values, axis=0)
    #     max_mean_group_train = np.max(mean_group_train, axis=0)
    #     min_mean_group_train = np.min(mean_group_train, axis=0)
    #     values_mean[runs[0].config.get("data_type") + " mean train"] = mean_group_train

    #     mean_group_test = np.mean(test_loss_values, axis=0)
    #     std_group_test = np.std(train_loss_values, axis=0)
    #     max_mean_group_test = np.max(mean_group_test, axis=0)
    #     min_mean_group_test = np.min(mean_group_test, axis=0)
    #     values_mean[runs[0].config.get("data_type") + " mean test"] = mean_group_test
    #     # print(mean_group)
    #     # print(std_group)
    #     # print(max_mean_group)
    #     # print(min_mean_group)
    #     # print(values_mean)
    # df = pd.DataFrame(values_mean)
    # train_columns = df.columns[df.columns.str.contains(pat="train")]
    # print(train_columns)
    # print(train_columns)
    # test_columns = df.columns[df.columns.str.contains(pat="test")]
    # df.plot(x="epoch", y=train_columns, kind="line", logy=True)
    # df.plot(x="epoch", y=test_columns, kind="line", logy=True)
    # plt.show()

    # if train_loss_values:
    #     # Convert list to a numpy array for calculations
    #     train_loss_values = np.array(train_loss_values)

    #     # Calculate mean and standard deviation
    #     mean = np.mean(train_loss_values)
    #     std_dev = np.std(train_loss_values)
    #     max_value = np.max(train_loss_values)

    #     print(
    #         f"For group {key}, mean train_loss: {mean}, std_dev train_loss: {std_dev}"
    #     )
    # else:
    #     print(f"For group {key}, no train_loss values found.")


def get_specific_values(data=None):
    if not data:
        data = pd.read_pickle("results.pickle")
    # print(data.loc[(data["reference"] == "fr-fr")])
    print(data)
    my_dict = {"identity_focus": None, "data_type": "log_quat_1"}
    # my_dict = {"reference": "fr-fr"}
    # my_dict = {"identity_focus": None}
    mask = np.logical_and.reduce(
        [pd.isnull(data[k]) if v is None else data[k] == v for k, v in my_dict.items()]
    )
    print(data.loc[mask])
    # print(data.loc[(data['column_name'] >= A) & (data['column_name'] <= B)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--new_collect_results",
        action="store_true",
        help="Force the model to focus on identity",
    )
    args = parser.parse_args()
    if args.new_collect_results:
        runs = get_runs()
        group_by = ["reference", "str_extra_input", "identity_focus", "data_type"]
        filtered_runs, grouped_runs = get_grouped_filtered_runs(
            runs,
            {
                # "reference": "fr-fr",
                "str_extra_input": None,
                "identity_focus": False,
                "data_dir_train": "data_t(5,20)_r(5,20)_combi_pNone_gNone",
            },
            group_by,
        )
        average_data = average_runs(grouped_runs)
        specific_df = get_specific_values(average_data)
    else:
        print("Using already collected runs !(These may be old)!")
        specific_df = get_specific_values()
    # history = run.history()
    #     print(history)
    #     for col in history:
    #         print(col)
    #     print(config)
    #     print(config.get("data_type"))
    #     print(history["Train loss"])
    #     exit()
