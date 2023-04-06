import pandas as pd

data = pd.read_csv("wandb_csv/wandb_export_2023-04-05T12_38_53.036+02_00.csv")
data_cleaned = data.drop(
    [
        "Created",
        "Runtime",
        "End Time",
        "Hostname",
        "ID",
        "Notes",
        "Updated",
        "Tags",
        "data_type",
        "test_sims",
        "train_sims",
    ],
    axis=1,
)
data_dir = data_cleaned.columns[1][10:-11]
print(data_dir)


final = pd.DataFrame()
final["name"] = data_cleaned["Name"]
final[data_cleaned.columns[1][:-11]] = data_cleaned[data_cleaned.columns[1]]
print(final)
final["name" == "pos"]["pizza"] = 1
print(final)
