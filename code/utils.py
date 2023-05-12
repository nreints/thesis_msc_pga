import argparse
import os
import random

import torch
import torch.utils.data as data

import wandb


def parse_args():
    """
    Parses the command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode_wandb",
        type=str,
        choices=["online", "offline", "disabled"],
        help="mode of wandb: online, offline, disabled",
        default="online",
    )
    parser.add_argument(
        "-train_dir",
        "--data_dir_train",
        type=str,
        help="directory of the train data",
        nargs="+",
        default="data_t(0,0)_r(5,20)_full_pNone_gNone",
    )
    parser.add_argument(
        "-test_dirs",
        "--data_dirs_test",
        type=str,
        help="directory/ies of the test data",
        nargs="+",
        default=[],
    )
    parser.add_argument("-l", "--loss", type=str, help="Loss type", default="L2")
    parser.add_argument("--data_type", type=str, help="Type of data", default="pos")
    parser.add_argument(
        "-i", "--iterations", type=int, help="Number of iterations", default=1
    )
    parser.add_argument(
        "-in_frames",
        "--input_frames",
        type=int,
        help="Number of input frames",
        default=10,
    )
    parser.add_argument(
        "-extra_input",
        type=str,
        choices=[
            "inertia_body",
            "size",
            "size_squared",
            "size_mass",
            "size_squared_mass",
        ],
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=0.001, help="Batch size"
    )
    return parser.parse_args()


def wandb_eval_string(data_dir_test, data_dir_train):
    """
    Returns the extra string to log the evaluation loss in WandB.

    Input:
        - data_dir_test: directory of the test data.
        - data_dir_train: directory of the train data.
    """
    if data_dir_test == data_dir_train[5:]:
        return ""
    else:
        return " " + data_dir_test[5:]


def nr_extra_input(extra_input_str):
    """
    Returns the number of extra input given the type of extra input.

    Input:
        - extra_input_str: string with the type of extra input.
    """
    n_extra_input = {
        "inertia_body": 3,
        "size": 3,
        "size_squared": 3,
        "size_mass": 4,
        "size_squared_mass": 4,
    }
    if extra_input_str:
        return n_extra_input[extra_input_str]
    else:
        return 0


def get_reference(data_type):
    """
    Returns string of how the data type is measured.
    With respect to the origin or with respect to the start frame.

    Input:
        - data_type: type of data (str).
    """
    if data_type[-3:] == "ori" or data_type == "pos":
        return "ori-fr"
    elif data_type[-4:] == "prev" or data_type[-1] == "1":
        return "fr-fr"
    else:
        return "start-fr"


def check_number_sims(data_dir_train, train_sims, data_dirs_test, test_sims):
    """
    Asserts whether the number of sims present in the train and test directory is enough.

    Input:
        - data_dir_train: data directory for the train data.
        - train_sims: IDs of the train simulations.
        - data_dirs_test: data directories for the test data.
        - test_sims: IDs of the test simulations.

    """
    assert len(os.listdir(data_dir_train)) >= max(
        train_sims
    ), "Not enough train simulations."
    for data_dir_test in data_dirs_test:
        assert len(os.listdir("data/" + data_dir_test)) >= max(
            test_sims
        ), f"Not enough test simulations in {data_dir_test}."
    print("Checked number of simulations in each data directory.")


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
    n_sims_train_total = 1000
    sims_train = range(0, n_sims_train_total)
    train_sims = random.sample(sims_train, int(0.8 * n_sims_train_total))
    test_sims = list(set(sims_train) - set(train_sims))
    check_number_sims(data_dir_train, train_sims, data_dirs_test, test_sims)
    print("Number of train simulations: ", len(train_sims))
    print("Number of test simulations: ", len(test_sims))
    return n_sims_train_total, train_sims, test_sims


def get_data_dirs(data_dir_train, data_dirs_test):
    """
    Returns the data directories for training and evaluating.

    Input:
        - data_dir_train: train directory (from command line arguments).

    Output:
        - data_train_dir: data directory for the train data.
        - data_dirs_test: data directories for the test data.
    """
    data_train_dir = " ".join(data_dir_train).replace('"', "")
    print(
        "with replace",
        [data_dir_test.replace('"', "") for data_dir_test in data_dirs_test],
    )
    print(f"Training on dataset: {data_train_dir}")
    print("test directories", data_dirs_test)
    print("with join", [" ".join(data_dir_test) for data_dir_test in data_dirs_test])
    # data_dirs_test = os.listdir("data")
    # if ".DS_Store" in data_dirs_test:
    #     data_dirs_test.remove(".DS_Store")
    data_dirs_test = [
        data_dir_test.replace('"', "") for data_dir_test in data_dirs_test
    ]
    data_dirs_test.insert(0, data_train_dir)
    data_dirs_test = list(set(data_dirs_test))
    print(f"Testing on {len(data_dirs_test)} datasets: {data_dirs_test}")
    return data_train_dir, data_dirs_test


def save_model(config, ndata_dict, model, normalize_extra_input):
    """
    Saves the model and other useful attributes.

    Input:
        - config: dictionary with (wandb) configuration.
        - ndata_dict: dictionary with number of datapoints per data type.
        - model: trained model.
        - normalize_extra_input: normalization factor to apply to the extra input.
    """
    model_dict = {
        "config": config,
        "data_dict": ndata_dict,
        "model": model.state_dict(),
        "normalize_extra_input": (
            config["str_extra_input"],
            config["extra_input_n"],
            normalize_extra_input,
        ),
    }
    os.makedirs("trained_models", exist_ok=True)
    os.makedirs(f"trained_models/{config['architecture']}", exist_ok=True)
    os.makedirs(
        f"trained_models/{config['architecture']}/{config['data_dir_train']}",
        exist_ok=True,
    )

    path_dir = f"trained_models/{config['architecture']}/{config['data_dir_train']}/'{config['data_type']}'_'{config['str_extra_input']}'.pth"
    torch.save(
        model_dict,
        path_dir,
    )

    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(path_dir)
    wandb.run.log_artifact(artifact)
    print("Saved model in ", path_dir)


def model_pipeline(
    hyperparameters,
    mode_wandb,
    losses,
    train_fn,
    device,
    dataset_class,
    model_class,
):
    """
    Enables WandB, creates model, trains model, and saves model.

    Input:
        - hyperparameters: dictionary with configuration for the model.
        - mode_wandb: mode of WandB.
        - losses: list of losses to evaluate on.
        - train_fn: specific train function of the model.
        - device: current device.
        - dataset_class: specific class with the dataset.
        - model_class: specific class of the model.

    Output:
        - model: trained model.
    """
    ndata_dict = {
        "pos": 24,
        "rot_mat": 12,
        "quat": 7,
        "log_quat": 7,
        "dual_quat": 8,
        "pos_diff": 24,
        "pos_diff_start": 24,
        "pos_norm": 24,
        "log_dualQ": 6,
    }
    loss_dict = {"L1": torch.nn.L1Loss, "L2": torch.nn.MSELoss}
    optimizer_dict = {"Adam": torch.optim.Adam}
    # tell wandb to get started
    with wandb.init(
        project="ThesisF", config=hyperparameters, mode=mode_wandb, tags=[str(device)]
    ):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        wandb.run.name = f"{config.architecture}/{config.data_type}/{config.iter}/{config.str_extra_input}/"

        # make the model, data, and optimization problem
        (
            model,
            train_loader,
            test_loaders,
            criterion,
            optimizer,
            normalize_extra_input,
        ) = make(
            config,
            ndata_dict,
            loss_dict,
            optimizer_dict,
            dataset_class,
            model_class,
            device,
        )
        print("Datatype:", config["data_type"])

        # and use them to train the model
        train_fn(
            model,
            optimizer,
            train_loader,
            test_loaders,
            criterion,
            config.epochs,
            config,
            losses,
            normalize_extra_input,
        )

        save_model(dict(config), ndata_dict, model, normalize_extra_input)

    return model


def make(
    config, ndata_dict, loss_dict, optimizer_dict, dataset_class, model_class, device
):
    """
    Makes dataloaders, model, criterion and optimizer.

    Input:
        - config: dictionary with (wandb) configuration.
        - ndata_dict: dictionary with number of datapoints per data type.
        - loss_dict: dictionary with loss name and corresponding pytorch functions.
        - optimizer_dict: dictionary with optimizer name and corresponding pytorch functions.
        - dataset_class: specific class with the dataset.
        - model_class: specific class of the model.
        - device: current device.

    Output:
        - model: model of correct class.
        - train_data_loader: dataloader for train data.
        - test_data_loaders: dataloader for test data.
        - criterion: loss function.
        - optimizer: optimizer function.
        - data_set_train.normalize_extra_input: normalization factor to apply to the extra input.
    """
    if config.data_type[-3:] == "ori":
        n_datapoints = ndata_dict[config.data_type[:-4]]
    elif config.data_type[-2:] == "_1":
        n_datapoints = ndata_dict[config.data_type[:-2]]
    elif config.data_type[-5:] == "_prev":
        n_datapoints = n_datapoints = ndata_dict[config.data_type[:-5]]
    else:
        n_datapoints = ndata_dict[config.data_type]
    # Make the data
    data_set_train = dataset_class(
        sims=config.train_sims,
        n_frames=config.n_frames,
        n_data=n_datapoints,
        data_type=config.data_type,
        dir="data/" + config.data_dir_train,
        extra_input=(config.str_extra_input, config.extra_input_n),
    )
    train_data_loader = data.DataLoader(
        data_set_train, batch_size=config.batch_size, shuffle=True
    )

    print("-- Finished Train Dataloader --")

    test_data_loaders = []

    for test_data_dir in config.data_dirs_test:
        data_set_test = dataset_class(
            sims=config.test_sims,
            n_frames=config.n_frames,
            n_data=n_datapoints,
            data_type=config.data_type,
            dir="data/" + test_data_dir,
            extra_input=(config.str_extra_input, config.extra_input_n),
        )
        test_data_loader = data.DataLoader(
            data_set_test, batch_size=config.batch_size, shuffle=True, drop_last=False
        )
        test_data_loaders += [test_data_loader]

    print("-- Finished Test Dataloader(s) --")

    # Make the model
    model = model_class(n_datapoints, config).to(device)
    print(model)

    # Make the loss and optimizer
    criterion = loss_dict[config.loss_type](reduction=config.loss_reduction_type)
    optimizer = optimizer_dict[config.optimizer](
        model.parameters(), lr=config.learning_rate
    )

    return (
        model,
        train_data_loader,
        test_data_loaders,
        criterion,
        optimizer,
        data_set_train.normalize_extra_input,
    )


def train_log(loss, epoch, loss_module, data_dir_train):
    """
    Logs the train loss to WandB.

    Input:
        - loss: loss of current epoch.
        - epoch: current epoch.
        - loss_module: type of loss.
        - data_dir_train: data directory for the train data.
    """
    wandb.log({"Epoch": epoch, "Train loss": loss}, step=epoch)
    print(
        f"\t Logging train Loss: {round(loss.item(), 10)} [{loss_module}: {data_dir_train}]"
    )


def eval_log(
    data_dir_test,
    data_dir_train,
    loss,
    epoch,
    loss_module,
):
    """
    Logs the test loss to WandB.

    Input:
        - data_dir_test: data directory for the test data.
        - data_dir_train: data directory for the train data.
        - loss: loss of current epoch.
        - epoch: current epoch.
        - loss_module: type of loss.
    """
    wandb_string = wandb_eval_string(data_dir_test, data_dir_train)
    wandb.log(
        {f"Test loss{wandb_string}": loss},
        step=epoch,
    )

    print(
        f"\t Logging test loss: {round(loss.item(), 10)} [{loss_module}: {data_dir_test[5:]}]"
    )
