import argparse
import os
import pickle5 as pickle
import time
from glob import glob
import json
import copy
import csv
import pandas as pd

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
from dtds.data.utils.transform import Transform
from torch.nn import functional as F
import time
from tqdm import tqdm
import datetime
from sklearn import preprocessing
from dtds.data.constants import CATEGORICAL, CONTINUOUS, ORDINAL
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)

def param_rrefs(module):
    """grabs remote references to the parameters of a module"""
    param_rrefs = []
    for param in module.parameters():
        param_rrefs.append(rpc.RRef(param))
    print(param_rrefs)
    return param_rrefs

def sum_of_layer(model_dicts, layer, weights_con_cat_combination):
    """
    Sum of parameters of one layer for all models
    """
    layer_sum = model_dicts[0][layer] * weights_con_cat_combination[0]
    for i in range(1, len(model_dicts)):
        layer_sum += model_dicts[i][layer] * weights_con_cat_combination[i]
    return layer_sum

def average_model(model_dicts, weights_con_cat_combination):
    """
    Average model by uniform weights
    """
    if len(model_dicts) == 1:
        return model_dicts[0]
    else:
        state_aggregate = model_dicts[0]
    for layer in state_aggregate:
        state_aggregate[layer] = sum_of_layer(model_dicts, layer, weights_con_cat_combination)

    return state_aggregate
    
    
    
    
def sum_of_layer_ordinary(model_dicts, layer):
    """
    Sum of parameters of one layer for all models
    """
    layer_sum = model_dicts[0][layer]
    for i in range(1, len(model_dicts)):
        layer_sum += model_dicts[i][layer]
    return layer_sum

def average_model_ordinary(model_dicts):
    """
    Average model by uniform weights
    """
    if len(model_dicts) == 1:
        return model_dicts[0]
    else:
        weights = 1/len(model_dicts)
        state_aggregate = model_dicts[0]
    for layer in state_aggregate:
        state_aggregate[layer] = weights*sum_of_layer_ordinary(model_dicts, layer)

    return state_aggregate
    
    
def softmax(vector):
	e = np.exp(vector)
	return e / e.sum()
	
def normal_average(vector):
    return vector / vector.sum()

def apply_activate(data, output_info):
    # ATTENTION: no adoption for 'tanh'
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
        else:
            assert 0

    return torch.cat(data_t, dim=1)

def sample(n, generator, cond_generator, transformer, batch_size, embedding_dim):
    generator.eval()

    output_info = transformer.output_info
    steps = n // batch_size + 1
    data = []
    for i in range(steps):
        mean = torch.zeros(batch_size, embedding_dim)
        std = mean + 1
        fakez = torch.normal(mean=mean, std=std).to(torch.device('cpu'))

        condvec = cond_generator.sample_zero(batch_size)
        if condvec is None:
            pass
        else:
            c1 = condvec
            c1 = torch.from_numpy(c1).to(torch.device('cpu'))
            fakez = torch.cat([fakez, c1], dim=1)

        fake = generator(fakez)
        fakeact = apply_activate(fake, output_info)
        data.append(fakeact.detach().cpu().numpy())

    data = np.concatenate(data, axis=0)
    data = data[:n]
    result = transformer.inverse_transform(data, None)
    
    while len(result) < n:
        data = []
        mean = torch.zeros(batch_size, embedding_dim)
        std = mean + 1
        fakez = torch.normal(mean=mean, std=std).to(torch.device('cpu'))

        condvec = cond_generator.sample_zero(batch_size)
        if condvec is None:
            pass
        else:
            c1 = condvec
            c1 = torch.from_numpy(c1).to(torch.device('cpu'))
            fakez = torch.cat([fakez, c1], dim=1)

        fake = generator(fakez)
        fakeact = apply_activate(fake, output_info)
        data.append(fakeact.detach().cpu().numpy())
        data = np.concatenate(data, axis=0)
        syn = transformer.inverse_transform(data, None)
        result = np.concatenate((result, syn))

    return result[0:n]

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
        # print("call in get_discriminator_weights: ", next(self.discriminator.parameters()).is_cuda)
        if next(self.discriminator.parameters()).is_cuda:
            return self.discriminator.cpu().state_dict()
        else:
            return self.discriminator.state_dict()

    def set_discriminator_weights(self, state_dict):
        # rint("call in set_discriminator_weights: ", self.device)
        self.discriminator.load_state_dict(state_dict)
        if self.device.type != 'cpu':
            self.discriminator.to(self.device)
        return True

    def set_generator_weights(self, state_dict):
        # print("call in set_generator_weights: ", self.device)
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
        # print("call in get_discriminator_weights: ")
        if next(self.discriminator.parameters()).is_cuda and next(self.generator.parameters()).is_cuda:
            return self.generator.cpu(), self.discriminator.cpu()
        else: 
            return self.generator, self.discriminator
class MDGANServer(ctgan.CTGANSynthesizer):
    """
    This is the class that encapsulates the functions that need to be run on the server side
    This is the main driver of the training procedure.
    """

    def __init__(self, client_rrefs, epochs, **kwargs):
        super(MDGANServer, self).__init__(**kwargs)
        # print("number of epochs in initialization: ", epochs)
        self.epochs = epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # keep a reference to the client
        self.client_rrefs = []
        for client_rref in client_rrefs:
            self.client_rrefs.append(client_rref)

    def save_model(self, filename):
        object_list = [self.generator, self.cond_generator, self.transformer, self.batch_size, self.embedding_dim]

        torch.save(object_list, f"models/model_list_rpc_{filename}.pt")
        
    def server_local_synthesizer_initialization(self):
        '''
        In general, server should not have data, the reason we load training data
        in server here, is because we will use the training data's category column frequency
        to construct the conditional vector for sampling synthetic data. When we use the generator 
        in server or client side, we don't need that step.
        The reason we have this step is only to make the synthetic data category column
        frequency is same as real data. This is for testing purpose.
        '''
        # every client should have a same transformer
        self.transformer = self.client_rrefs[0].remote().get_transformer().to_here()
        # train is same data as data/raw/Intrusion_train.csv, just in array format
        train = np.load("models/Intrusion_train.npz")['train']
        # print("in save model train after: ", train.shape)
        train = self.transformer.transform(train)
        self.cond_generator = ctgan.Cond(train, self.transformer.output_info) 
              
    def sample_data(self, index):
        synthesis = sample(40000, self.generator, self.cond_generator, self.transformer, self.batch_size, self.embedding_dim)
        meta_path = 'models/Intrusion.json'
        # synthesis.to_csv("fake_before_inverse.csv", index=False)
        df, time_stamp, column_names = Transform.inverse(
            synthesis, meta_path , self.label_encoder, epochs=300, fname='{}_generate'.format('Intrusion')
        )
        output_name = "Intrusion_result/{}_synthesis_epoch_{}.csv".format('Intrusion', index)
        pd.DataFrame(df).to_csv(output_name, index=False)

    def uniform_meta_category(self):
        meta_list = []
        label_encoders_data = []
        for client_rref in self.client_rrefs:
            meta_client = client_rref.remote().get_meta().to_here()
            meta_list.append(meta_client)

        meta_base = copy.deepcopy(meta_list[0])
        # first found out how many categorical column
        cat_counter = 0
        for column_idx, column in enumerate(meta_base["columns"]):
            if column["type"] == 'categorical':
                cat_counter += 1
        # for each bit, is the sum of JS divergence of all categorical columns for one worker
        self.distribution_similarity_vector = []
        for index in range(len(self.client_rrefs)):
            self.distribution_similarity_vector.append(np.zeros(cat_counter))

        category_cursor = 0
        for column_idx, column in enumerate(meta_base["columns"]):
            if column["type"] == 'categorical':
                category_dict = {}
                for idx_, meta_element in enumerate(meta_list):
                    for element_key in meta_element['columns'][column_idx]['i2s']:
                        if element_key in category_dict.keys():
                            category_dict[element_key] += meta_element['columns'][column_idx]['i2s'][element_key]
                        else:
                            category_dict[element_key] = meta_element['columns'][column_idx]['i2s'][element_key]

                meta_base['columns'][column_idx]['i2s'] = list({k: v for k, v in sorted(category_dict.items(), key=lambda item: item[1], reverse=True)}.keys())
                label_encoder = preprocessing.LabelEncoder()
                label_encoder.fit(meta_base['columns'][column_idx]['i2s'])
                label_encoders_data.append(label_encoder)

                vec_global = np.zeros(len(category_dict.keys())) 
                for key in category_dict:
                    vec_global[label_encoder.transform(np.array([key]))[0]] = category_dict[key]
                temp_distribution_vector = []
                for index in range(len(self.client_rrefs)):
                    temp_distribution_vector.append(np.zeros(len(category_dict.keys()))) 

                for idx_, meta_element in enumerate(meta_list):
                    for key in meta_element['columns'][column_idx]['i2s']:
                        temp_distribution_vector[idx_][label_encoder.transform(np.array([key]))[0]] = meta_element['columns'][column_idx]['i2s'][key]

                for index in range(len(self.client_rrefs)):
                    self.distribution_similarity_vector[index][category_cursor] = distance.jensenshannon(vec_global, temp_distribution_vector[index])
                category_cursor += 1

        # print("weights for aggregation for categorical columns before normalization",self.distribution_similarity_vector)
        temp_sum_distance = np.sum(self.distribution_similarity_vector, axis = 0)

        for index in range(len(self.client_rrefs)):
            for ind in range(len(temp_sum_distance)):
                if temp_sum_distance[ind] != 0: # in that case, all the similarity values are 0, it happens when there is only one participant    
                    self.distribution_similarity_vector[index][ind] = self.distribution_similarity_vector[index][ind]/temp_sum_distance[ind]
        # print("weights for aggregation for categorical columns", self.distribution_similarity_vector)
        
        zero_index = []
        for idx_, element in enumerate(temp_sum_distance):
            if element == 0:
                zero_index.append(idx_)

        for index in zero_index:
            for i in range(len(self.client_rrefs)):
                self.distribution_similarity_vector[i][index] = 1.0/len(self.client_rrefs)
        # print("weights for aggregation for categorical columns after post-processing", self.distribution_similarity_vector)
      

        for client_rref in self.client_rrefs:
            time_stamp = str(datetime.datetime.now().timestamp()).replace(".", "")
            # here need to be rpc_sync() to make sure in the client side, all the results are updated before entering next calls
            client_rref.rpc_sync().register_meta(meta_base, time_stamp, label_encoders_data)


        # this block of code is to construct the label_encoder format as the same as it is in the standalone setting
        # this will be easy to use for prediction usage
        label_encoder_dic = []
        label_encoder_cursor = 0
        for column_idx, column in enumerate(meta_base["columns"]):
            if column["type"] == 'categorical':
                current_label_encoder = dict()
                current_label_encoder['column_name'] = column["column_name"]
                current_label_encoder['label_encoder'] = label_encoders_data[label_encoder_cursor]
                label_encoder_cursor += 1
                label_encoder_dic.append(current_label_encoder)

        with open("{}/label_encoders_{}.pickle".format('models', 'Intrusion'), "wb") as f:
            pickle.dump(label_encoder_dic, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        self.label_encoder = label_encoder_dic
        with open('{}/{}.json'.format('models', 'Intrusion'), 'w') as f:
            json.dump(meta_base, f, sort_keys=True, indent=4, separators=(',', ': '), cls=NumpyEncoder)

        # print(meta_base)


    def uniform_continuous_gmm(self):
        meta_list = []
        model_list = []
        components_list = []
        rows_number_list = []
        n_clusters = 10
        eps = 0.005
        for client_rref in self.client_rrefs:
            model, component, meta, rows = client_rref.remote().get_transformer_information().to_here()
            meta_list.append(meta)
            model_list.append(model)
            components_list.append(component)
            rows_number_list.append(rows)

        print("total data number: ", np.sum(rows_number_list))
        n_sample = np.sum(rows_number_list)
        self.model_weights_by_number = [float(x)/np.sum(rows_number_list) for x in rows_number_list]
        print("aggregation weights by data number: ", self.model_weights_by_number)
        model_aggregate = []
        components_aggregate = []


        # first found out how many continuous column
        continuous_counter = 0
        for column_idx, info in enumerate(meta_list[0]):
            if info["type"] == CONTINUOUS:
                continuous_counter += 1
        # for each bit, is the sum of WD of all continuous columns for one worker
        self.distribution_similarity_vector_continuous = []
        for index in range(len(self.client_rrefs)):
            self.distribution_similarity_vector_continuous.append(np.zeros(continuous_counter))

        continuous_cursor = 0
        for id_, info in enumerate(meta_list[0]):
            if info["type"] == CONTINUOUS:
                # print("continuous: ", info["name"])
                gm = BayesianGaussianMixture(
                    n_components=n_clusters,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=0.001,
                    n_init=1,
                )
                gmm_sample = []
                for i in range(len(model_list)):
                    # sample totally n_sample data from all gmm
                    gmm_sample.append(model_list[i][id_].sample(int(n_sample*self.model_weights_by_number[i]))[0].reshape([1,-1])[0])
                gmm_sample_array = np.concatenate(gmm_sample)

                # following block is to calculate wasserstein distance
                for index in range(len(self.client_rrefs)):
                    self.distribution_similarity_vector_continuous[index][continuous_cursor] = wasserstein_distance(gmm_sample_array, gmm_sample[index])


                # fit on all sampled data
                gm.fit(gmm_sample_array.reshape([-1,1]))
                model_aggregate.append(gm)
                comp = gm.weights_ > eps
                components_aggregate.append(comp)
                continuous_cursor += 1

            else:
                model_aggregate.append(None)
                components_aggregate.append(None)

        # print("weights for aggregation for continuous columns before normalization", self.distribution_similarity_vector_continuous)
        temp_sum_distance = np.sum(self.distribution_similarity_vector_continuous, axis = 0)

         # when temp_sum_distance == 0, it can be that there is only one participant     
        for index in range(len(self.client_rrefs)):
            for ind in range(len(temp_sum_distance)):
                if temp_sum_distance[ind] != 0:
                    self.distribution_similarity_vector_continuous[index][ind] = self.distribution_similarity_vector_continuous[index][ind]/temp_sum_distance[ind]
        # print("weights for aggregation for continuous columns", self.distribution_similarity_vector_continuous)

        for client_rref in self.client_rrefs:
            # here need to be rpc_sync() to make sure in the client side, model and components are indeed updated
            client_rref.rpc_sync().register_transformer_information(model_aggregate, components_aggregate)
        
    def calculate_final_weights_for_aggregation(self):
        """
        self.weights_con_cat_combination is the final weights for each workers
        """
        weights_continuous = np.sum(self.distribution_similarity_vector_continuous, axis = 1)
        # print("weights_continuous: ", weights_continuous)
        weights_categorical = np.sum(self.distribution_similarity_vector, axis = 1)
        # print("weights_categorical: ", weights_categorical)
        self.weights_con_cat_combination = np.sum([weights_continuous, weights_categorical], axis = 0)
        # print("weights_con_cat_combination: ", self.weights_con_cat_combination)
        sum_weights = np.sum(self.weights_con_cat_combination)
        # print("sum_weights: ", sum_weights)
        for i in range(self.weights_con_cat_combination.shape[0]):
            # print("final weights before times ratio of number of data: ", i, 1-self.weights_con_cat_combination[i]/sum_weights)
            self.weights_con_cat_combination[i] = (1-self.weights_con_cat_combination[i]/sum_weights)*self.model_weights_by_number[i]
        self.weights_con_cat_combination = softmax(self.weights_con_cat_combination)
        print("final aggregation weights", self.weights_con_cat_combination)

    def fit(self):
    
        # After initialization, generator and discrminator should have same structure across all participants
        # For simplicity of the code. Server just get that from first client. 
        self.generator, self.discriminator = self.client_rrefs[0].remote().get_models().to_here()
        accumulated_time_list = []
        time_training_list = []
        time_aggregation_list = []
        time_distribution_list = []
        for i in tqdm(range(self.epochs)):
            time_in = time.time()
            model_dicts_list = []
            model_dicts_G = []
            model_dicts_D = []
            for client_rref in self.client_rrefs:
                model_dicts = client_rref.rpc_async().train_model(1)
                model_dicts_list.append(model_dicts)

            for model_dict in model_dicts_list:
                model_dict_g, model_dict_d  = model_dict.wait()   
                model_dicts_G.append(model_dict_g)
                model_dicts_D.append(model_dict_d)
            time_training_list.append(time.time() - time_in)
            time_start = time.time() 
            average_model_G = {}
            average_model_G = average_model(model_dicts_G, self.weights_con_cat_combination)                
            self.generator.load_state_dict(average_model_G)
            
            average_model_D = {}
            average_model_D = average_model(model_dicts_D, self.weights_con_cat_combination)
            self.discriminator.load_state_dict(average_model_D)
            
            time_aggregation_list.append(time.time() - time_start)
            time_start1 = time.time()
            
            self.sample_data(i)
            for client_rref in self.client_rrefs:
                client_rref.rpc_sync().set_discriminator_weights(average_model_D)
                client_rref.rpc_sync().set_generator_weights(average_model_G)                    
            time_distribution_list.append(time.time() - time_start1)
            accumulated_time_list.append(time.time() - time_in)
            
        with open('timestamp_experiment.csv','w') as result_file:
            wr = csv.writer(result_file, dialect='excel')
            wr.writerows(np.array(accumulated_time_list).reshape(-1,1))


    def refit_local_transformer(self):
        for client_rref in self.client_rrefs:
            # here need to be rpc_sync() to make sure in the client side, transformer is indeed updated
            client_rref.rpc_sync().refit_transformer()        


def run(rank, einterval, world_size, ip, port, name, datapath, selected_variables, categorical_list, nonnegative_list, date_dic,
target_column, problem_type, epochs, report):
    # set environment information
    os.environ["MASTER_ADDR"] = ip
    os.environ["MASTER_PORT"] = str(port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    print("number of training epochs: ", epochs)
    print("world size: ", world_size, f"tcp://{ip}:{port}")
    if rank == 0:  # this is run only on the server side
        print("server")
        rpc.init_rpc(
            "server",
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.PROCESS_GROUP,
            rpc_backend_options=rpc.ProcessGroupRpcBackendOptions(
                num_send_recv_threads=8, rpc_timeout=600, init_method=f"tcp://{ip}:{port}"
            ),
        )
        clients = []
        for worker in range(world_size-1):
            clients.append(rpc.remote("client"+str(worker+1), MDGANClient, kwargs=dict(datapath=datapath,
            selected_variables=selected_variables, categorical_list=categorical_list, nonnegative_list=nonnegative_list, date_dic=date_dic,
            target_column=target_column, problem_type = problem_type, epochs = epochs)))
            print("register remote client"+str(worker+1), clients[0])

        synthesizer = MDGANServer(clients, epochs=epochs)
        print("Begin initialization for categorical columns")
        synthesizer.uniform_meta_category()
        print("Begin initialization for continuous columns")
        synthesizer.uniform_continuous_gmm()
        synthesizer.refit_local_transformer()
        print("Calculate aggregation weights for participants")
        synthesizer.calculate_final_weights_for_aggregation()
        synthesizer.server_local_synthesizer_initialization()
        print("Finishing initialization")
        synthesizer.fit()


    elif rank != 0:
        rpc.init_rpc(
            "client"+str(rank),
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                num_worker_threads=8, rpc_timeout=600, init_method=f"tcp://{ip}:{port}"
            ),
        )
        print("client"+str(rank)+" is joining")


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
    parser.add_argument("-E_interval", type=int, default=1)
    parser.add_argument("-world_size", type=int, default=2)
    parser.add_argument("-report", action="store_true")
    
    
    #parser.add_argument("-target_column", type=str, default='')
    parser.add_argument("-problem_type", type=str, default='')
    
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

    args = parser.parse_args()

    if args.rank is not None:
        # run with a specified rank (need to start up another process with the opposite rank elsewhere)
        run(
            rank=args.rank,
            einterval=args.E_interval,
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
