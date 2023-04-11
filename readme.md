# MSc AI Thesis

### How to run:
#### 1. Install the required packages:
please run `pip install -r requirements.txt` or `pip3 install -r requirements.txt` before running the code.

#### 2. Create data:
To generate 100 simulations each of 500 frames for a rod-like cuboid (ratio [1:1:10]) run and a initial angular velocity:<br>
`python create_data.py -symmetry="semi" -n_sims=100 -n_frames=500 -l_min 0 -l_max 0 -a_min 2 -a_max 5`<br>


To generate 100 simulations each of 500 frames for a tennis-like cuboid (ratio [1:3:10]) run and a initial linear velocity:<br>
`python create_data.py -symmetry="tennis" -n_sims=100 -n_frames=500 -l_min 4 -l_max 10 -a_min 0 -a_max 0`<br>

etc.

#### 3. Train a network:
1. FCNN: currently trained and evaluated on 1 dataset, and evaluated on specified other datasets. Example on how to run: <br>
`python fcnn.py -data_dir_train="data_t(0, 0)_r(2, 5)_none_pNone_gNone" -iterations=1 -data_type="pos" -loss="L1"`
2. LSTM: currently trained on and evaluated 1 dataset, and evaluated on specified other datasets.<br>
`python lstm.py -data_dir_train="data_t(0, 0)_r(2, 5)_none_pNone_gNone" -iterations=1 -data_type="pos" -loss="L1"`
3. GRU: currently trained on and evaluated 1 dataset, and evaluated on specified other datasets.<br>
`python gru.py -data_dir_train="data_t(0, 0)_r(2, 5)_none_pNone_gNone" -iterations=1 -data_type="pos" -loss="L1"`


#### 4. Plots:

##### 1. Plot prediction of a model:<br>
`python plot_data.py --prediction -data_dir="data_t(0, 0)_r(2, 5)_full_pNone_gNone"`

##### 2. Plot different data types:<br>
`python plot_data.py -data_dir="data_t(0, 0)_r(6, 8)_full_pNone_gNone"`
