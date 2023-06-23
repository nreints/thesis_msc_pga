import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_runs():
    api = wandb.Api()

    # Define your project
    project = "nreints/ThesisFinal"

    runs = api.runs(project)
    print(f"There are {len(runs)} runs in ThesisFinal.")
    return runs


def get_filtered_runs(runs, filters):
    filtered_runs = []
    for run in runs[:50]:
        config = run.config

        # If it satisfies all filters, save it
        if all([config.get(filter) == filters[filter] for filter in filters]):
            filtered_runs += [run]
        # for filter in filters:
        #     if config.get(filter) == filters[filter]:
        # filtered_runs += [run]
        # key = [config.get(filter) for filter in filters]
        # print(key)
    print(f"{len(filtered_runs)} runs satisfy the filters")
    return filtered_runs


def group_runs(runs, groups):
    grouped_runs = {}
    for run in runs:
        config = run.config

        # group_thing = config.get(groups[0])
        # group2_thing = config.get(groups[1])
        # print(type(group_thing))
        # print(type(group2_thing))
        # key_try = (group_thing, group2_thing)
        # print(key_try, type(key_try))

        key = tuple([config.get(group) for group in groups])
        # print(config.get(groups[0]))
        # print(keys)
        # for key in key_gen:
        #     print(type(key))
        # print(type(key_gen))
        # exit()

        if key in grouped_runs:
            grouped_runs[key].append(run)
        else:
            grouped_runs[key] = [run]
    return grouped_runs


def average_runs(group_dict):
    values_mean = {}
    for key, runs in (group_dict).items():
        train_loss_values = np.zeros((len(runs), runs[0].config.get("epochs")))
        test_loss_values = np.zeros((len(runs), runs[0].config.get("epochs")))
        values_mean["epoch"] = range(runs[0].config.get("epochs"))
        for i, run in enumerate(runs):
            history = run.history()
            # train_loss_values.extend(history.get("Train loss", []))
            # test_loss_values.extend(history.get("Train loss", []))
            train_loss_values[i] = history.get("Train loss")
            test_loss_values[i] = history.get(
                "Test loss t(5,20)_r(5,20)_combi_pNone_gNone"
            )

        mean_group_train = np.mean(train_loss_values, axis=0)
        std_group_train = np.std(train_loss_values, axis=0)
        max_mean_group_train = np.max(mean_group_train, axis=0)
        min_mean_group_train = np.min(mean_group_train, axis=0)
        values_mean[runs[0].config.get("data_type") + " mean train"] = mean_group_train

        mean_group_test = np.mean(train_loss_values, axis=0)
        std_group_test = np.std(train_loss_values, axis=0)
        max_mean_group_test = np.max(mean_group_test, axis=0)
        min_mean_group_test = np.min(mean_group_test, axis=0)
        values_mean[runs[0].config.get("data_type") + " mean test"] = mean_group_test
        # print(mean_group)
        # print(std_group)
        # print(max_mean_group)
        # print(min_mean_group)
        # print(values_mean)
    df = pd.DataFrame(values_mean)
    train_columns = df.columns[df.columns.str.contains(pat="train")]
    print(train_columns)
    test_columns = df.columns[df.columns.str.contains(pat="test")]
    df.plot(x="epoch", y=train_columns, kind="line", logy=True)
    df.plot(x="epoch", y=test_columns, kind="line", logy=True)
    plt.show()

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


if __name__ == "__main__":
    runs = get_runs()
    filtered_runs = get_filtered_runs(
        runs,
        {
            "focus_identity": False,
            "data_dir_train": "data_t(5,20)_r(5,20)_combi_pNone_gNone",
            "reference": "fr-fr",
        },
    )
    group_by = ["data_type"]
    grouped_runs = group_runs(filtered_runs, group_by)

    average_data = average_runs(grouped_runs)
    # history = run.history()
    #     print(history)
    #     for col in history:
    #         print(col)
    #     print(config)
    #     print(config.get("data_type"))
    #     print(history["Train loss"])
    #     exit()
