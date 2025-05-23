import time

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential
from torch.nn import functional as F
from tqdm import tqdm

from dtds.synthesizers.base import BaseSynthesizer
from dtds.features.transformers import BGMTransformer


class Discriminator(Module):
    def __init__(self, input_dim, dis_dims, pack=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pack
        self.pack = pack
        self.packdim = dim
        seq = []
        for item in list(dis_dims):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item
        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def forward(self, input):
        assert input.size()[0] % self.pack == 0
        return self.seq(input.view(-1, self.packdim))


class Residual(Module):
    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)


class Generator(Module):
    def __init__(self, embedding_dim, gen_dims, data_dim):
        """
        gen_dims = (256, 256), list(gen_dims) ->  [256, 256]
        """
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(gen_dims):
            seq += [Residual(dim, item)]
            dim += item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data


def apply_activate(data, output_info):
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == "tanh":
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == "softmax":
            ed = st + item[0]
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
        else:
            assert 0

    return torch.cat(data_t, dim=1)


def random_choice_prob_index(a, axis=1):
    """
    r is the random number, so a.cumsum give the cumulative sum along row, , (e.g, [0.3 0,5 0,9 1 1 1 1])
    if r = 0.6, (a.cumsum(axis=axis) > r) = [false false true true true true true]
    [false false true true true true true].argmax() is 2, which returns the index of the first true
    """
    r = np.expand_dims(np.random.rand(a.shape[1 - axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)


def maximum_interval(output_info):
    max_interval = 0
    for item in output_info:
        max_interval = max(max_interval, item[0])
    return max_interval


class Cond(object):
    def __init__(self, data, output_info):
        self.model = []
        st = 0
        counter = 0
        for item in output_info:
            if item[1] == "tanh":
                st += item[0]
                continue
            elif item[1] == "softmax":

                ed = st + item[0]
                counter += 1
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                st = ed
            else:
                assert 0


        assert st == data.shape[1]

        self.interval = []
        self.n_col = 0 
        self.n_opt = 0  
        st = 0
        self.p = np.zeros((counter, maximum_interval(output_info)))
        for item in output_info:
            if item[1] == "tanh":
                st += item[0]
                continue
            elif item[1] == "softmax":
                ed = st + item[0]
                tmp = np.sum(data[:, st:ed], axis=0) 
                tmp = np.log(tmp + 1)  
                tmp = tmp / np.sum(tmp)
                self.p[self.n_col, : item[0]] = tmp
                self.interval.append((self.n_opt, item[0]))
                self.n_opt += item[0]
                self.n_col += 1
                st = ed
            else:
                assert 0

        self.interval = np.asarray(self.interval)

    def sample(self, batch):
        if self.n_col == 0:
            return None
        batch = batch
        idx = np.random.choice(
            np.arange(self.n_col), batch
        ) 
        vec1 = np.zeros((batch, self.n_opt), dtype="float32")
        mask1 = np.zeros((batch, self.n_col), dtype="float32")
        mask1[np.arange(batch), idx] = 1  
        opt1prime = random_choice_prob_index(self.p[idx])
        for i in np.arange(batch):
            vec1[i, self.interval[idx[i], 0] + opt1prime[i]] = 1
            
        return vec1, mask1, idx, opt1prime

    def sample_zero(self, batch):
        if self.n_col == 0:
            return None
        vec = np.zeros((batch, self.n_opt), dtype='float32')
        idx = np.random.choice(np.arange(self.n_col), batch)
        for i in range(batch):
            col = idx[i]  
            pick = int(np.random.choice(self.model[col]))
            vec[i, pick + self.interval[col, 0]] = 1
        return vec

def cond_loss(data, output_info, c, m):
    loss = []
    st = 0
    st_c = 0
    for item in output_info:
        if item[1] == "tanh":
            st += item[0]

        elif item[1] == "softmax":

            ed = st + item[0]
            ed_c = st_c + item[0]
            tmp = F.cross_entropy(data[:, st:ed], torch.argmax(c[:, st_c:ed_c], dim=1), reduction="none")
            loss.append(tmp)
            st = ed
            st_c = ed_c

        else:
            assert 0
    loss = torch.stack(loss, dim=1)
    return (loss * m).sum() / data.size()[0]


class Sampler(object):
    """docstring for Sampler."""

    def __init__(self, data, output_info):
        super(Sampler, self).__init__()
        self.data = data
        self.model = []
        self.n = len(data)
        st = 0
        for item in output_info:
            if item[1] == "tanh":
                st += item[0]
            elif item[1] == "softmax":
                ed = st + item[0]
                tmp = []
                for j in range(item[0]):
                    tmp.append(np.nonzero(data[:, st + j])[0])           
                self.model.append(tmp)
                st = ed
            else:
                assert 0

        assert st == data.shape[1]

    def sample(self, n, col, opt):
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]
        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))
        return self.data[idx]


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1)).view(val.size(0), 1)
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high
    return res


def calc_gradient_penalty(netD, real_data, fake_data, device="cpu", pac=10, lambda_=10):

    alpha = torch.rand(real_data.size(0), 1, device=device)
    interpolates = slerp(alpha, real_data, fake_data)
    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size(), device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    # http://blog.gqylpy.com/gqy/14030/
    # grad_outputs=torch.ones(disc_interpolates.size(), device=device): 就是对计算出的gradient的加权，可以理解为weights

    gradient_penalty = ((gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1) ** 2).mean() * lambda_
    return gradient_penalty


ave_grads = {"G": [], "D": []}
layers = {"G": [], "D": []}


def update_grad_flow(named_parameters, net):
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("TkAgg")

    ave_grads[net].append([])
    layers[net] = []
    for n, p in named_parameters:
        if p.requires_grad:
            layers[net].append(n)
            try:
                ave_grads[net][-1].append(p.grad.abs().mean())
            except:
                ave_grads[net][-1].append(0)


def plot_grad_flow(save_path):
    from matplotlib import pyplot as plt
    import matplotlib.pylab as pl

    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    colors = pl.cm.winter(np.linspace(0, 1, len(ave_grads["G"])))
    for p, net in enumerate(["D", "G"]):
        for i in range(len(ave_grads["G"])):
            ax[p].plot(layers[net], ave_grads[net][i], alpha=0.5, color=colors[i])
        ax[p].hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="grey")
        ax[p].set_xticks(range(0, len(ave_grads), 1), layers)
        ax[p].set_xlim(xmin=0, xmax=len(ave_grads))
        ax[p].tick_params(axis="x", rotation=30)
        ax[p].set_xlabel(f"Layers {net}")

    ax[1].legend(
        [
            plt.Line2D([0], [0], color=colors[0], lw=4),
            plt.Line2D([0], [0], color=colors[-1], lw=4),
        ],
        ["Start", "End"],
    )
    ax[0].set_ylabel("Average gradient")
    plt.suptitle("Gradient flow")
    plt.savefig(save_path + "/grad_flow.png")


class CTGANSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        embedding_dim=128,
        gen_dim=(
            256,
            256,
        ),
        dis_dim=(
            256,
            256,
        ),
        l2scale=1e-6,
        batch_size=500,
        epochs=3,
    ):

        self.__name__ = "CTGANSynth"
        self.embedding_dim = embedding_dim
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, train_data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.transformer = BGMTransformer()
        self.transformer.fit(train_data, categorical_columns, ordinal_columns)
        self.out_info = self.transformer.output_info

        train_data = self.transformer.transform(train_data)

        data_sampler = Sampler(train_data, self.out_info)

        data_dim = self.transformer.output_dim
        self.cond_generator = Cond(train_data, self.out_info)
        steps_per_epoch = len(train_data) // self.batch_size

        self.generator = Generator(self.embedding_dim + self.cond_generator.n_opt, self.gen_dim, data_dim).to(
            self.device
        )

        discriminator = Discriminator(data_dim + self.cond_generator.n_opt, self.dis_dim).to(self.device)

        optimizerG = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=self.l2scale)
        optimizerD = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.9))

        assert self.batch_size % 2 == 0
        mean = torch.zeros(self.batch_size, self.embedding_dim, device=self.device)
        std = mean + 1

        for i in range(self.epochs):
            time_in = time.time()
            for _ in range(steps_per_epoch):
                fakez = torch.normal(mean=mean, std=std)
                condvec = self.cond_generator.sample(self.batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                    real = data_sampler.sample(self.batch_size, col, opt)
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c2 = c1[perm]
                fake = self.generator(fakez)
                fakeact = apply_activate(fake, self.out_info)

                real = torch.from_numpy(real.astype("float32")).to(self.device)

                if c1 is not None:
                    fake_cat = torch.cat([fakeact, c1], dim=1)
                    real_cat = torch.cat([real, c2], dim=1)
                else:
                    real_cat = real
                    fake_cat = fake

                y_fake = discriminator(fake_cat)
                y_real = discriminator(real_cat)

                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                pen = calc_gradient_penalty(discriminator, real_cat, fake_cat, self.device)

                optimizerD.zero_grad()
                pen.backward(retain_graph=True)
                loss_d.backward()
                optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self.cond_generator.sample(self.batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self.device)
                    m1 = torch.from_numpy(m1).to(self.device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self.generator(fakez)
                fakeact = apply_activate(fake, self.out_info)

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = cond_loss(fake, self.out_info, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad()
                loss_g.backward()
                # update_grad_flow(self.generator.named_parameters(), "G")
                optimizerG.step()
            print(
                f"EPOCH {i}:   loss_d:{loss_d.item():>6.2f}   loss_g:{loss_g.item():>6.2f}   time taken: {time.time() - time_in:.2f} sec"
            )

        # plot_grad_flow()

    def sample(self, n):
        self.generator.eval()

        output_info = self.out_info
        steps = n // self.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self.device)

            condvec = self.cond_generator.sample_zero(self.batch_size)
            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self.device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self.generator(fakez)
            fakeact = apply_activate(fake, output_info)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        result = self.transformer.inverse_transform(data, None)
        print("first round generated result length: ", len(result))
        while len(result) < n:
            data = []
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self.device)

            condvec = self.cond_generator.sample_zero(self.batch_size)
            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self.device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self.generator(fakez)
            fakeact = apply_activate(fake, output_info)
            data.append(fakeact.detach().cpu().numpy())
            data = np.concatenate(data, axis=0)
            syn = self.transformer.inverse_transform(data, None)
            result = np.concatenate((result, syn))
            print("length of generated result: ", len(result))
        return result[0:n]