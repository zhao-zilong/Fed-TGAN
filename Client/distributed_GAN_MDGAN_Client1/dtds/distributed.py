import argparse
import os
import pickle5 as pickle
import time
from glob import glob
import json

import dtds.synthesizers.ctgan as ctgan
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.optim
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from sklearn.mixture import BayesianGaussianMixture
from dtds.data.load import load_datapath, prepare_data, encode_data_with_meta_labelencoder
from dtds.features.transformers import BGM_CTGAN_Transformer, decode_train_data
from dtds.data.utils.file_generator import FileGenerator
import time
from tqdm import tqdm
import datetime
from sklearn import preprocessing
from dtds.data.constants import CATEGORICAL, CONTINUOUS, ORDINAL
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
def param_rrefs(module):
    """grabs remote references to the parameters of a module"""
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(rpc.RRef(param))
    print(param_rrefs)
    return param_rrefs

def sum_of_layer(model_dicts, layer):
    """
    Sum of parameters of one layer for all models
    """
    layer_sum = model_dicts[0][layer]
    for i in range(1, len(model_dicts)):
        layer_sum += model_dicts[i][layer]
    return layer_sum

def average_model(model_dicts):
    """
    Average model by uniform weights
    """
    if len(model_dicts) == 1:
        return model_dicts[0]
    else:
        weights = 1/len(model_dicts)
        state_aggregate = model_dicts[0]
    for layer in state_aggregate:
        state_aggregate[layer] = weights*sum_of_layer(model_dicts, layer)

    return state_aggregate

class MDGANClient(ctgan.CTGANSynthesizer):
    """
    This is the class that encapsulates the functions that need to be run on the client side
    Despite the name, this source code only needs to reside on the server and will be executed via RPC.
    """

    def __init__(self, datapath, selected_variables, categorical_list, nonnegative_list, date_dic, target_column, problem_type, epochs, **kwargs):
        super(MDGANClient, self).__init__(**kwargs)
        # RPC doesn't support sending GPU tensors, need to be explictly bring back to host memory fist
        # this would require some more overrides (e.g. the forward() functions) so is not implemented for now
        #self.device = torch.device("cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.datapath = datapath
        self.selected_variables = selected_variables
        self.categorical_list = categorical_list
        self.nonnegative_list = nonnegative_list
        self.date_dic = date_dic
        self.epochs = epochs
        self.target_column = target_column
        self.problem_type = problem_type
        self.meta_file = prepare_data(self.datapath, self.selected_variables, self.categorical_list, self.nonnegative_list,
        self.date_dic, self.target_column, self.problem_type)
        self.filename = self.datapath.split('/')[-1].split('.')[0]
        self.time_train_d = []
        self.time_loss_g = []
    def send_client_refs(self):
        """Send a reference to the discriminator (for future RPC calls) and the conditioner, transformer and steps/epoch"""
        return rpc.RRef(self.discriminator), self.cond_generator, self.transformer, self.steps_per_epoch

    def register_G(self, G_rref):
        """Receive a reference to the generator (for future RPC calls)"""
        self.G_rref = G_rref

    def get_discriminator_weights(self):
        if next(self.discriminator.parameters()).is_cuda:
            return self.discriminator.cpu().state_dict()
        else:
            return self.discriminator.state_dict()

    def set_discriminator_weights(self, state_dict):
        self.discriminator.load_state_dict(state_dict)
        if self.device.type != 'cpu':
            self.discriminator.to(self.device)
        return True

    def set_generator_weights(self, state_dict):
        self.generator.load_state_dict(state_dict)
        if self.device.type != 'cpu':
            self.generator.to(self.device)
        return True



    def reset_on_cuda(self):
        if self.device.type != 'cpu':
            self.discriminator.to(self.device)        

    def register_meta(self, meta, timestamp, label_encoder):
        self.meta_file = meta
        self.timestamp = timestamp
        self.label_encoder = label_encoder

    def get_meta(self):
        return self.meta_file

    def encode_data(self):
        self.train, self.cat_cols, self.ord_cols = encode_data_with_meta_labelencoder(self.datapath,
        self.meta_file, self.label_encoder, self.selected_variables, self.categorical_list, self.nonnegative_list,
        self.date_dic, self.target_column, self.problem_type, self.timestamp)
        self.rows = self.train.shape[0]

    def get_transformer_information(self):
        self.train, self.cat_cols, self.ord_cols = encode_data_with_meta_labelencoder(self.datapath, self.meta_file, self.label_encoder, self.selected_variables, self.categorical_list, self.nonnegative_list,
        self.date_dic, self.target_column, self.problem_type, self.timestamp)
        self.rows = self.train.shape[0]
        self.transformer = BGM_CTGAN_Transformer()
        self.transformer.fit(self.train, self.cat_cols, self.ord_cols)
        self.model, self.components, self.meta_transformer = self.transformer.get_information()
        return self.model, self.components, self.meta_transformer, self.rows

    def register_transformer_information(self, model, components):
        self.model = model
        self.components = components

    def refit_transformer(self):
        self.transformer.refit(self.train, self.meta_file, self.label_encoder, self.cat_cols, self.ord_cols, self.model, self.components)
        train = self.transformer.transform(self.train)
        self.out_info = self.transformer.output_info
        self.data_dim = self.transformer.output_dim

        self.sampler = ctgan.Sampler(train, self.out_info)

        self.cond_generator = ctgan.Cond(train, self.out_info)
        self.steps_per_epoch = len(self.train) // self.batch_size
        self.generator = ctgan.Generator(
            self.embedding_dim + self.cond_generator.n_opt,
            self.gen_dim,
            self.transformer.output_dim).to(self.device)
        self.discriminator = ctgan.Discriminator(
            input_dim=self.data_dim + self.cond_generator.n_opt, dis_dims=self.dis_dim
        )    
        if self.device.type != 'cpu':
            self.discriminator.to(self.device)
            self.generator.to(self.device)                  
        self.optG = optim.Adam(
            self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=self.l2scale)
        self.optD = optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))
        
  

    def get_transformer(self):
        return self.transformer

    def get_cond_generator(self):
        return self.cond_generator


    def train_model(self, number_of_epoch = 1):
        print("New round training model on client")
        if self.device.type != 'cpu' and not next(self.generator.parameters()).is_cuda:
            self.generator.to(self.device)
        if self.device.type != 'cpu' and not next(self.discriminator.parameters()).is_cuda:
            self.discriminator.to(self.device)
        for epoch_index in range(number_of_epoch):
            for _ in range(self.steps_per_epoch):
                mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
                std = mean + 1
                fakez = torch.normal(mean=mean, std=std)
                condvec = self.cond_generator.sample(self.batch_size)
                # process conditioning vector
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = self.sampler.sample(self.batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = self.sampler.sample(self.batch_size, col[perm], opt[perm])
                    c2 = c1[perm]

                # send fakez back to the server anget_discriminator_weightsd receive the fake data
                # (maybe there's a way to avoid this back and forth by moving part of the previous if statement to the server?)
                # fake = self.G_rref.remote().forward(fakez).to_here()
                # print("debug: ", next(self.generator.parameters()).is_cuda, fakez.is_cuda)
                fake = self.generator(fakez)
                fakeact = ctgan.apply_activate(fake, self.out_info)

                # forward through the local discriminator
                real = torch.from_numpy(real.astype("float32")).to(self.device)
                if c1 is not None:
                    fake_cat = torch.cat([fakeact, c1], dim=1)
                    real_cat = torch.cat([real, c2], dim=1)
                else:
                    real_cat = real
                    fake_cat = fake
                y_fake = self.discriminator(fake_cat)
                y_real = self.discriminator(real_cat)

                # send losses back to server
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                pen = ctgan.calc_gradient_penalty(self.discriminator, real_cat, fake_cat, self.device)
                self.optD.zero_grad()
                #pen.backward(retain_graph=True)
                loss_d_sum = loss_d + pen
                loss_d_sum.backward()
                self.optD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self.cond_generator.sample(self.batch_size)
                # process conditioning vector
                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                # send fakez back to the server and receive the fake data
                # (maybe there's a way to avoid this back and forth by moving part of the previous if statement to the server?)
                # fake = self.G_rref.remote().forward(fakez).to_here()
                fake = self.generator(fakez)
                fakeact = ctgan.apply_activate(fake, self.out_info)

                # forward through the local discriminator
                if c1 is not None:
                    y_fake = self.discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = self.discriminator(fakeact)
                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = ctgan.cond_loss(fake, self.out_info, c1, m1)

                # send loss back to server
                loss_g = -torch.mean(y_fake) + cross_entropy

                self.optG.zero_grad()
                loss_g.backward()
                self.optG.step()
        if self.device.type != 'cpu':
            return self.generator.cpu().state_dict(), self.discriminator.cpu().state_dict()
        else:
            return self.generator.state_dict(), self.discriminator.state_dict()


    # here is the same schema as MD-GAN paper which trains D on client side
    def train_D(self):
        if self.device.type != 'cpu' and not next(self.generator.parameters()).is_cuda:
            self.generator.to(self.device)
        if self.device.type != 'cpu' and not next(self.discriminator.parameters()).is_cuda:
            self.discriminator.to(self.device)
        for _ in range(self.steps_per_epoch):
            mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std)
            condvec = self.cond_generator.sample(self.batch_size)
            # process conditioning vector
            if condvec is None:
                c1, m1, col, opt = None, None, None, None
                real = self.sampler.sample(self.batch_size, col, opt)
            else:
                c1, m1, col, opt = condvec
                c1 = torch.from_numpy(c1).to(self.device)
                m1 = torch.from_numpy(m1).to(self.device)
                fakez = torch.cat([fakez, c1], dim=1)

                perm = np.arange(self.batch_size)
                np.random.shuffle(perm)
                real = self.sampler.sample(self.batch_size, col[perm], opt[perm])
                #real = torch.tensor(real, requires_grad=True, device=self.device)
                c2 = c1[perm]

            # send fakez back to the server and receive the fake data
            # (maybe there's a way to avoid this back and forth by moving part of the previous if statement to the server?)
            time_stamp_start = time.time()
            if self.device.type != 'cpu':
                fake = self.G_rref.remote().forward(fakez.cpu()).to_here()
                fake = fake.to(device=self.device)
            else:
                fake = self.G_rref.remote().forward(fakez).to_here()

            fakeact = ctgan.apply_activate(fake, self.out_info)
            self.time_train_d.append(time.time() - time_stamp_start)
            
            # forward through the local discriminator         
            real = torch.from_numpy(real.astype("float32")).to(self.device)
            if c1 is not None:
                fake_cat = torch.cat([fakeact, c1], dim=1)
                real_cat = torch.cat([real, c2], dim=1)
            else:
                real_cat = real
                fake_cat = fake
            y_fake = self.discriminator(fake_cat)
            y_real = self.discriminator(real_cat)
            # send losses back to server
            loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
            pen = ctgan.calc_gradient_penalty(self.discriminator, real_cat, fake_cat, self.device)
            self.optD.zero_grad()
            pen.backward(retain_graph=True)
            loss_d.backward()
            self.optD.step()

        if self.device.type != 'cpu':
            return loss_d.cpu(), pen.cpu()
        else:
            return loss_d, pen


    def loss_G(self):
        loss_g_list = []
        for _ in range(self.steps_per_epoch):
            mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std)
            condvec = self.cond_generator.sample(self.batch_size)
            # process conditioning vector
            if condvec is None:
                c1, m1, col, opt = None, None, None, None
            else:
                c1, m1, col, opt = condvec
                c1 = torch.from_numpy(c1).to(self.device)
                m1 = torch.from_numpy(m1).to(self.device)
                fakez = torch.cat([fakez, c1], dim=1)

            # send fakez back to the server and receive the fake data
            # (maybe there's a way to avoid this back and forth by moving part of the previous if statement to the server?)
            time_stamp_start = time.time()
            if self.device.type != 'cpu':
                fake = self.G_rref.remote().forward(fakez.cpu()).to_here()
                fake = fake.to(self.device)
            else:
                fake = self.G_rref.remote().forward(fakez).to_here()
            fakeact = ctgan.apply_activate(fake, self.out_info)
            self.time_loss_g.append(time.time() - time_stamp_start)
            # forward through the local discriminator
            if c1 is not None:
                y_fake = self.discriminator(torch.cat([fakeact, c1], dim=1))
            else:
                y_fake = self.discriminator(fakeact)
            if condvec is None:
                cross_entropy = 0
            else:
                cross_entropy = ctgan.cond_loss(fake, self.out_info, c1, m1)

            # send loss back to server
            loss_g = -torch.mean(y_fake) + cross_entropy
            if self.device.type != 'cpu':
                loss_g_list.append(loss_g.cpu())
            else:
                loss_g_list.append(loss_g)
        return sum(loss_g_list)

    def save_time_stamp(self):
        with open('time_train_d.csv','w') as result_file:
            wr = csv.writer(result_file, dialect='excel')
            wr.writerows(np.array(self.time_train_d).reshape(-1,1))
            
        with open('time_loss_g.csv','w') as result_file:
            wr = csv.writer(result_file, dialect='excel')
            wr.writerows(np.array(self.time_loss_g).reshape(-1,1))    
        

    def get_models(self):
        if next(self.discriminator.parameters()).is_cuda and next(self.generator.parameters()).is_cuda:
            return self.generator.cpu(), self.discriminator.cpu()
        else: 
            return self.generator, self.discriminator

def run(rank, world_size, ip, port, name, datapath, selected_variables, categorical_list, nonnegative_list, date_dic,
target_column, problem_type, epochs, report):
    # set environment information
    os.environ["MASTER_ADDR"] = ip
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    print("world size:", world_size,f"tcp://{ip}:{port}")
    print("client")
    rpc.init_rpc(
        "client"+str(rank),
        rank=rank,
        world_size=world_size,
        backend=rpc.BackendType.PROCESS_GROUP,
        rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(
            num_send_recv_threads=8, rpc_timeout=600, init_method=f"tcp://{ip}:{port}"
        ),
    )
    print("Client"+str(rank)+" is joining")

    rpc.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-rank", type=int, default=None)
    parser.add_argument("-ip", type=str, default="127.0.0.1")
    parser.add_argument("-port", type=int, default=7788)
    parser.add_argument("-name", type=str, default="Intrusion_train")
    parser.add_argument(
        "-datapath", type=str, default="data/raw/Intrusion_train.csv"
    )
    parser.add_argument("-epochs", type=int, default=10)
    parser.add_argument("-world_size", type=int, default=2)
    parser.add_argument("-report", action="store_true")

    parser.add_argument("-target_column", type=str, default='class')
    parser.add_argument("-selected_variables", type=list, default=['duration', 'protocol_type', 'service', 'flag', 'src_bytes',
       'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
       'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
       'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
       'num_access_files', 'num_outbound_cmds', 'is_host_login',
       'is_guest_login', 'count', 'srv_count', 'serror_rate',
       'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
       'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
       'dst_host_srv_count', 'dst_host_same_srv_rate',
       'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
       'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
       'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
       'dst_host_srv_rerror_rate', 'class'])
    parser.add_argument("-categorical_list", type=list, default=[ 'protocol_type', 'service', 'flag', 'land', 'wrong_fragment', 'urgent', 'hot',
       'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
       'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
       'num_access_files', 'num_outbound_cmds', 'is_host_login',
       'is_guest_login', 'class'])
    parser.add_argument("-nonnegative_list", type=list, default=['dst_bytes','src_bytes'])
    parser.add_argument("-date_dic", type=dict, default={})
    
    parser.add_argument("-problem_type", type=str, default='')

    args = parser.parse_args()

    if args.rank is not None:
        # run with a specified rank (need to start up another process with the opposite rank elsewhere)
        run(
            rank=args.rank,
            world_size=args.world_size,
            ip=args.ip,
            port=args.port,
            name=args.name,
            datapath=args.datapath,
            selected_variables = args.selected_variables,
            categorical_list = args.categorical_list,
            nonnegative_list = args.nonnegative_list,
            date_dic = args.date_dic,
            target_column = args.target_column,
            problem_type = args.problem_type,
            epochs=args.epochs,
            report=args.report,

        )
    else:
        # run both client and server locally
        mp.spawn(
            run,
            args=(
                2,
                args.ip,
                args.port,
                args.name,
                args.datapath,
                args.epochs,
                args.report,
            ),
            nprocs=2,
            join=True,
        )
