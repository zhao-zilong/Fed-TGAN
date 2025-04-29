# FED-TGAN: Federated Learning Framework for Synthesizing Tabular Data 

This repo is the testbed for paper ***FED-TGAN: Federated Learning Framework for Synthesizing Tabular Data*** and ***GDTS: GAN-based distributed tabular synthesizer***. The code contains two parts: `Server` and `Client`. The server is the federator in federated learning (FL). The clients are the participants. The testbed is a distributed framework realised on Pytorch RPC. The server and client can be placed on different computers, as long as they can communicate via internet. I will first introduce how to train the FED-TGAN algorithm (using Intrusion dataset as a demo). Then explain how to evaluate the results (i.e., statistical similarity and ML utility) in the same way as we showed in the paper.

## Training process

### Server
Since we use Pytorch RPC as our communication method, here are some parameters we need to provide to organize the federated learning group. For example, to run the demo with Intrusion dataset with 2 clients, we can first enter folder `Server` and type the following command:
```
python3 -m dtds.distributed -ip 192.168.1.76 -rank 0 -epoch 500 -world_size 3 -datapath "data/raw/Intrusion_train.csv"
```
There are several parameters in this command. `ip` is the ip of the **server**. `rank` indicates the rank of the current runner. It's important to set `rank` as **0** for running server. `epoch` is self-explained, it is the training epoch number. `world_size` is the number of clients plus one server. For instance, here we set `world_size` to 3, it means we will have two clients. `datapath` is the path of training dataset that in the **client** side, which means all the client should have the training data in the same relative path. 

### Client
Under `Client` folder, we already provided two client folders: `distributed_GAN_MDGAN_Client0` and `distributed_GAN_MDGAN_Client1`. Actually, there is no difference between the two folders. If anyone wants to add more clients to the FL system, just copy and paste this folder to your local machine. One thing to notice is that clients are not necessarily located in the same machine as the server, our demo works as long as there is network connection between the server and clients.

For clients to join the FL, under both `distributed_GAN_MDGAN_Client0` and `distributed_GAN_MDGAN_Client1` folders, run:
```
python -m dtds.distributed -ip 192.168.1.76 -rank 1 -world_size 3
```
and
```
python -m dtds.distributed -ip 192.168.1.76 -rank 2 -world_size 3
```
We can observe that for two clients, the only difference is that clients need to self-claim a unique rank for themself. And for `N` clients, their ranks should start from **1** to **N** without repetitions. And the `ip` is the server's ip. We can also check that there is file `data/raw/Intrusion_train.csv` under `distributed_GAN_MDGAN_Client0` and `distributed_GAN_MDGAN_Client1`.

### Possible Training error
On the Linux system where we implement our experiment, one possible error is 
```
RuntimeError: ECONNREFUSED: connection refused
```
This is because we use Gloo as backend. By default, Gloo backend will try to find the right network interface to use. But sometimes it's not correct. In that case, we need to override it with:
```
export GLOO_SOCKET_IFNAME=eth0
```
`eth0` is your network interface name, you may need to change it. Use ```ifconfig``` in the terminal to check the name.

### Output
- During the training, under `Server/Intrusion_result/`, it will save the generated synthetic data at the end of each training epoch. For instance, if we set `epoch` to 500, then there will have datasets named from `Intrusion_synthesis_epoch_0.csv` to `Intrusion_synthesis_epoch_499.csv` in `Server/Intrusion_result/` folder.
- A file named `timestamp_experiment.csv` will be generated after successfully finishing the training process under `Server` folder. 

## Evaluation of the result

### Statistical Similarity
For statistical similarity, going to `Server` folder and use:
```
python similarity_analysis.py -nepoch 500
```
This script will generate a file `Intrusion_statistical_similarity_analysis.csv` under `Server` folder. The content looks like that:

| Epoch_No. | Avg_JSD | Avg_WD | time_stamp | 
| ----------- | ----------- | ----------- | ----------- |
| 0 | 0.19 | 0.08 | 24.26|
| 1 | 0.082 | 0.04 |48.46|

The unit of time_stamp is `second`.

### ML Utility
For ML utility, going to `Server` folder and use:
```
python utility_analysis.py -train_path 'data/raw/Intrusion_train.csv' -test_path 'data/raw/Intrusion_test.csv' -synthetic_path 'Intrusion_result/Intrusion_synthesis_epoch_499.csv'
```
In general, you do not need to change the path for `train_path` and `test_path` for this demo. But for `synthetic_path`, you should check the existance of your synthetic data. In this demo, we assume you train the model 500 epochs.
 
In the terminal, after evaluation, you will see the output like:
```
difference in f1-score: 0.0849535625139106
```

## Citation
You can cite our framework by following bibtex:
```
@article{zhao2021fed,
  title={Fed-TGAN: Federated Learning Framework for Synthesizing Tabular Data},
  author={Zhao, Zilong and Birke, Robert and Kunar, Aditya and Chen, Lydia Y},
  journal={arXiv preprint arXiv:2108.07927},
  year={2021}
}
```
or
```
@inproceedings{zhao2023gdts,
  title={GDTS: GAN-based distributed tabular synthesizer},
  author={Zhao, Zilong and Birke, Robert and Chen, Lydia Y},
  booktitle={2023 IEEE 16th International Conference on Cloud Computing (CLOUD)},
  pages={570--576},
  year={2023},
  organization={IEEE}
}
```
