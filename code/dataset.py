import os
import pickle
import time

import torch
import torch.utils.data as data


class RecurrentDataset(data.Dataset):
    def __init__(self, sims, n_frames, n_data, data_type, dir, extra_input):
        """
        Inputs:
            - sims; simulation IDs to use in this dataset
            - n_frames; number of input frames
            - n_data; number of datapoints given the data_type
            - data_type; type of the data
            - dir; directory where the data is stored
            - extra_input; tuple
                - extra_input[0]; type of extra input
                - extra_input[1]; number of extra input values
        """
        super().__init__()
        self.n_frames_perentry = n_frames
        self.n_datap_perframe = n_data
        self.sims = sims
        self.data_type = data_type
        self.dir = dir
        self.extra_input = extra_input
        self.collect_data()

    def collect_data(self):
        start_time = time.time()
        count = 0
        for i in self.sims:
            with open(f"{self.dir}/sim_{i}.pickle", "rb") as f:
                data_all = pickle.load(f)["data"]
                data = torch.FloatTensor(data_all[self.data_type])
                if count == 0:
                    data_per_sim = len(data) - (self.n_frames_perentry + 1)
                    len_data = len(self.sims) * data_per_sim
                    self.target = torch.zeros(
                        (len_data, self.n_frames_perentry, self.n_datap_perframe)
                    )
                    self.target_pos = torch.zeros(
                        (len_data, self.n_frames_perentry, 24)
                    )
                    self.start_pos = torch.zeros((len_data, self.n_frames_perentry, 24))
                    self.data = torch.zeros(
                        len_data, self.n_frames_perentry, self.n_datap_perframe
                    )
                    self.extra_input_data = torch.zeros((len_data, self.extra_input[1]))
                    self.xpos_start = torch.zeros((len_data, self.n_frames_perentry, 3))
                for frame in range(data_per_sim):
                    train_end = frame + self.n_frames_perentry
                    self.data[count] = data[frame:train_end].reshape(
                        -1, self.n_datap_perframe
                    )
                    self.target[count] = data[frame + 1 : train_end + 1].reshape(
                        -1, self.n_datap_perframe
                    )
                    self.target_pos[count] = torch.FloatTensor(
                        data_all["pos"][frame + 1 : train_end + 1]
                    ).flatten(start_dim=1)
                    if self.data_type[-1] == "1" or self.data_type[-4:] == "prev":
                        self.start_pos[count] = torch.FloatTensor(
                            data_all["pos"][frame:train_end]
                        ).flatten(start_dim=1)
                        self.xpos_start[count] = torch.FloatTensor(
                            data_all["xpos"][frame:train_end]
                        ).flatten(start_dim=1)
                    else:
                        if self.data_type[-3:] != "ori":
                            self.start_pos[count] = (
                                torch.FloatTensor(data_all["pos"][0])[None, :, :]
                                .flatten(start_dim=1)
                                .repeat(1, self.n_frames_perentry, 1)
                            )
                        else:
                            self.start_pos[count] = (
                                torch.FloatTensor(data_all["start"])
                                .flatten(start_dim=1)[:, None, :]
                                .repeat(1, self.n_frames_perentry, 1)
                            )
                        self.xpos_start[count] = torch.FloatTensor(
                            data_all["xpos_start"]
                        )[None, None, :].repeat(1, self.n_frames_perentry, 1)

                    if self.extra_input[1] != 0:
                        extra_input_values = torch.FloatTensor(
                            data_all["inertia_body"]
                        )
                        self.extra_input_data[count] = extra_input_values
                    count += 1

        self.normalize_extra_input = torch.mean(
            torch.norm(self.extra_input_data, dim=1)
        )
        print(f"The dataloader for {self.dir} took {time.time() - start_time} seconds.")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        data_point = self.data[idx]
        data_target = self.target[idx]
        data_target_pos = self.target_pos[idx]
        data_start = self.start_pos[idx]
        extra_input_data = self.extra_input_data[idx]
        start_xpos = self.xpos_start[idx]
        return (
            data_point,
            data_target,
            data_target_pos,
            data_start,
            extra_input_data,
            start_xpos,
        )


class NonRecurrentDataset(data.Dataset):
    def __init__(
        self,
        sims,
        n_frames,
        n_data,
        data_type,
        dir,
        extra_input,
    ):
        """
        Inputs:
            - sims; simulation IDs to use in this dataset
            - n_frames; number of input frames
            - n_data; number of datapoints given the data_type
            - data_type; type of the data
            - dir; directory where the data is stored
            - extra_input; tuple
                - extra_input[0]; type of extra input
                - extra_input[1]; number of extra input values
        """
        super().__init__()
        self.n_frames_perentry = n_frames
        self.n_datap_perframe = n_data
        self.sims = sims
        self.data_type = data_type
        self.dir = dir
        self.extra_input = extra_input
        self.collect_data()

    def collect_data(self):
        start_time = time.time()
        count = 0
        # Loop through all simulations
        for i in self.sims:
            with open(f"{self.dir}/sim_{i}.pickle", "rb") as f:
                data_all = pickle.load(f)["data"]
                # Collect data from data_type
                data = torch.FloatTensor(data_all[self.data_type])
                pos_data = torch.FloatTensor(data_all["pos"])
                # Add data and targets
                if count == 0:
                    data_per_sim = len(data) - (self.n_frames_perentry + 1)
                    len_data = len(self.sims) * data_per_sim
                    self.data = torch.zeros(
                        len_data,
                        self.n_frames_perentry * self.n_datap_perframe
                        + self.extra_input[1],
                    )
                    self.target = torch.zeros((len_data, self.n_datap_perframe))
                    self.target_pos = torch.zeros((len_data, 24))
                    self.start_pos = torch.zeros_like(self.target_pos)
                    self.xpos_start = torch.zeros((len_data, 3))
                for frame in range(data_per_sim):
                    # Always save the start position for converting
                    train_end = frame + self.n_frames_perentry
                    if self.data_type[-1] == "1" or self.data_type[-4:] == "prev":
                        self.start_pos[count] = torch.FloatTensor(
                            data_all["pos"][train_end]
                        ).flatten()
                        self.xpos_start[count] = torch.FloatTensor(
                            data_all["xpos"][train_end]
                        ).flatten()
                    else:
                        if self.data_type[-3:] != "ori":
                            self.start_pos[count] = torch.FloatTensor(
                                data_all["pos"][0].flatten()
                            )
                        else:
                            self.start_pos[count] = torch.FloatTensor(
                                data_all["start"].flatten()
                            )
                        self.xpos_start[count] = torch.FloatTensor(
                            data_all["xpos_start"].flatten()
                        )

                    if self.extra_input[1] != 0:
                        extra_input_values = torch.FloatTensor(
                            data_all["inertia_body"]
                        )
                        self.data[count, -self.extra_input[1] :] = extra_input_values
                        self.data[count, : -self.extra_input[1]] = data[
                            frame:train_end
                        ].flatten()
                    else:  # Extra input
                        self.data[count] = data[frame:train_end].flatten()

                    self.target[count] = data[train_end + 1].flatten()

                    self.target_pos[count] = pos_data[train_end + 1].flatten()
                    count += 1

        # self.mean = torch.mean(self.data)
        # self.std = torch.std(self.data)
        # self.normalized_data = (self.data - self.mean) / self.std

        self.normalize_extra_input = torch.mean(
            torch.norm(self.data[:, -self.extra_input[1] :], dim=1)
        )

        assert (
            self.normalize_extra_input != 0
        ), f"The normalization of the extra input is zero. This leads to zero-division."
        # self.data[:, -self.extra_input[1] :] = self.extra_input_data
        print(f"The dataloader for {self.dir} took {time.time() - start_time} seconds.")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        data_point = self.data[idx]
        data_target = self.target[idx]
        data_start = self.start_pos[idx]
        data_pos_target = self.target_pos[idx]
        start_xpos = self.xpos_start[idx]
        return data_point, data_target, data_start, data_pos_target, start_xpos
