# MSc AI Thesis

### How to run:
#### To install the required packages:
please run `pip install -r requirements.txt` or `pip3 install -r requirements.txt` before running the code

#### Create data:
To generate 100 simulations each of 500 frames for a rod-like cuboid (ratio [1,1,10]) run and a initial angular velocity:
`python create_data.py -symmetry="semi" -n_sims=100 -n_frames=500 -l_min 0 -l_max 0 -a_min 2 -a_max 5`
To generate 100 simulations each of 500 frames for a tennis-like cuboid (ratio [1,3,10]) run and a initial linear velocity:
`python create_data.py -symmetry="tennis" -n_sims=100 -n_frames=500 -l_min 4 -l_max 10 -a_min 0 -a_max 0`

etc.

#### Train a network:
1. FCNN: currently trained on 1 dataset, and evaluated on all other present datasets<br>

2. LSTM: currently trained on 1 dataset, and evaluated on all other present datasets<br>

3. GRU: currently trained on 1 dataset, and evaluated on all other present datasets<br>


#### Plots:

#### Plot prediction of a model:

#### Plot different data types:
