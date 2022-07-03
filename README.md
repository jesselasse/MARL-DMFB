# MARL-DMFB
A cooperative multi-agent reinforcement learning framework for droplet routing for droplet routing in DMFB

## Train model
python main.py --chip_size=10 --drop_num=4 

training data will be saved in "TrainResult/vdn/10by10-4d0b/"

trained model will be saved in "model/vdn/10by10-4d0b/"

## Evaluate model in health mode
python main.py --evaluate --load_model --chip_size=50 --drop_num=4 --evaluate_epoch=100

This will evaluate the performance of the model: "model/vdn/50by50-4d0b/rnn_net_params.pkl" and "model/vdn/50by50-4d0b/vdn_net_params.pkl"

## Evaluate model with electrodes degrade
python evaDegre.py --chip_size=10 --drop_num=4 --evaluate_epoch=40

This will evaluate the performance of the model: "model/vdn/10by10-4d0b/rnn_net_params.pkl" and "model/vdn/10by10-4d0b/vdn_net_params.pkl"

The data will be saved in "DgreData/10by10-4d0b"

## Parameters
You can find more usages or change the parameters of the algorithm in the file "common/arguments.py"
