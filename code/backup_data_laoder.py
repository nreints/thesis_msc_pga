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
                self.n_frames_perentry * self.n_datap_perframe + self.extra_input[1],
            )
            self.extra_input_data = torch.zeros(len_data, self.extra_input[1])
            self.target = torch.zeros((len_data, self.n_datap_perframe))
            self.target_pos = torch.zeros((len_data, 24))
            self.start_pos = torch.zeros_like(self.target_pos)
        for frame in range(data_per_sim):
            # Always save the start position for converting
            self.start_pos[count] = pos_data[0].flatten()
            train_end = frame + self.n_frames_perentry
            if self.extra_input[1] != 0:
                extra_input_values = torch.FloatTensor(data_all[self.extra_input[0]])
                self.extra_input_data[count] = extra_input_values
            else:
                self.data[count] = data[frame:train_end].flatten()
            self.target[count] = data[train_end + 1].flatten()

            self.target_pos[count] = pos_data[train_end + 1].flatten()
            count += 1

self.mean = torch.mean(self.data)
self.std = torch.std(self.data)
self.normalized_data = (self.data - self.mean) / self.std
if self.norm_extra_input:
    # self.extra_input_data = (
    #     self.extra_input_data - torch.mean(self.extra_input_data, 0)
    # ) / torch.std(self.extra_input_data, 0)
    self.extra_input_data /= 100000000
self.data[:, -self.extra_input[1] :] = self.extra_input_data
print(time.time() - start_time)


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
                self.n_frames_perentry * self.n_datap_perframe + self.extra_input[1],
            )
            self.target = torch.zeros((len_data, self.n_datap_perframe))
            self.target_pos = torch.zeros((len_data, 24))
            self.start_pos = torch.zeros_like(self.target_pos)
        for frame in range(data_per_sim):
            # Always save the start position for converting
            self.start_pos[count] = pos_data[0].flatten()
            train_end = frame + self.n_frames_perentry
            if self.extra_input[1] != 0:
                # TODO
                inertia = torch.FloatTensor(data_all[self.extra_input[0]]) / 100000000

                # inertia = np.array([1, 2, 3])
                self.data[count] = torch.cat((data[frame:train_end].flatten(), inertia))
            else:
                self.data[count] = data[frame:train_end].flatten()
            self.target[count] = data[train_end + 1].flatten()

            self.target_pos[count] = pos_data[train_end + 1].flatten()
            count += 1

self.mean = torch.mean(self.data)
self.std = torch.std(self.data)
self.normalized_data = (self.data - self.mean) / self.std
print(time.time() - start_time)
