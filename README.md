# MARL-DMFB
A cooperative multi-agent reinforcement learning framework for droplet routing for droplet routing in DMFB

## Train model
python train.py dmfb --drop_num=4 --fov=9

the training data will be saved in "data-dmfb/TrainResult/vdn/fov9/10by10-4d0b/"

the trained model will be saved in "data-dmfb/model/vdn/fov9/4d0b/

## Evaluate model in health mode
python main.py --evaluate --load_model --chip_size=50 --drop_num=4 --evaluate_epoch=100
python evaluate.py dmfb --drop_num=4 --chip_size=20 --evaluate_task=100 --show
This will evaluate the performance of the model: "data-dmfb/model/vdn/fov9/4d0b/rnn_net_params.pkl" and "data-dmfb/model/vdn/fov9/4d0b/vdn_net_params.pkl"

## Evaluate model with electrodes degrade
python evaDegre.py dmfb --evaluate_task = 20 --fov=9 --drop_num=4

This will evaluate the performance of the model: "data-dmfb/model/vdn/fov9/4d0b/rnn_net_params.pkl" and "data-dmfb/model/vdn/fov9/4d0b/vdn_net_params.pkl"

The data will be saved in "data-dmfb/DgreData/10by10-4d0b"

## For MEDA
just change 'dmfb' to 'meda' in the commands: i.e., training command is
python train.py meda --drop_num=4

Then all data are saved under fold "data-meda/"

## Parameters
You can find more usages or change the parameters of the algorithm in the file "common/arguments.py"
