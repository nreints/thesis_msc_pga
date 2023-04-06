import os
import torch
import wandb

def save_model(config, ndata_dict, model, normalize_extra_input):
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
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/fcnn", exist_ok=True)
    os.makedirs(f"models/fcnn/{config['data_dir_train']}", exist_ok=True)

    path_dir = f"models/fcnn/{config['data_dir_train']}/'{config['data_type']}'_'{config['str_extra_input']}'.pth"
    torch.save(
        model_dict,
        path_dir,
    )

    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(path_dir)
    wandb.run.log_artifact(artifact)
    print("Saved model in ", path_dir)


def model_pipeline(
    hyperparameters, ndata_dict, loss_dict, optimizer_dict, mode_wandb, losses
):
    # tell wandb to get started
    with wandb.init(
        project="test", config=hyperparameters, mode=mode_wandb, tags=[str(device)]
    ):
        # access all HPs through wandb.config, so logging matches execution!
        config_wandb = wandb.config
        wandb.run.name = f"{config_wandb.architecture}/{config_wandb.data_type}/{config_wandb.iter}/{config_wandb.str_extra_input}/"

        # make the model, data, and optimization problem
        (
            model,
            train_loader,
            test_loaders,
            criterion,
            optimizer,
            normalize_extra_input,
        ) = make(
            config_wandb,
            ndata_dict,
            loss_dict,
            optimizer_dict,
        )
        print("Datatype:", config_wandb["data_type"])

        # and use them to train the model
        train_model(
            model,
            optimizer,
            train_loader,
            test_loaders,
            criterion,
            config_wandb.epochs,
            config_wandb,
            losses,
            normalize_extra_input,
        )

        # # and test its final performance
        # eval_model(
        #     model,
        #     test_loaders,
        #     criterion,
        #     config_wandb,
        #     config_wandb.epochs,
        #     losses,
        #     normalize_extra_input,
        # )
        save_model(config, ndata_dict, model, normalize_extra_input)

    return model


def make(config, ndata_dict, loss_dict, optimizer_dict):
    if config.data_type[-3:] == "ori":
        n_datapoints = ndata_dict[config.data_type[:-4]]
    else:
        n_datapoints = ndata_dict[config.data_type]
    # Make the data
    data_set_train = MyDataset(
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
        data_set_test = MyDataset(
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
    model = fcnn(n_datapoints, config).to(device)
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
