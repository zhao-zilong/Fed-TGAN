import datetime
import json

import numpy as np
import pandas as pd
from dtds.data.constants import CATEGORICAL, CONTINUOUS, ORDINAL
from dtds.data.utils.date import Date
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import KBinsDiscretizer



class Transformer:
    @staticmethod
    def get_metadata(data, categorical_columns=tuple(), ordinal_columns=tuple()):
        meta = []

        df = pd.DataFrame(data)
        for index in df:
            column = df[index]

            if index in categorical_columns:
                mapper = column.value_counts().index.tolist()
                meta.append({"name": index, "type": CATEGORICAL, "size": len(mapper), "i2s": mapper})
            elif index in ordinal_columns:
                value_count = list(dict(column.value_counts()).items())
                value_count = sorted(value_count, key=lambda x: -x[1])
                mapper = list(map(lambda x: x[0], value_count))
                meta.append({"name": index, "type": ORDINAL, "size": len(mapper), "i2s": mapper})
            # we accept non-negative type, then it will be seen as continuous
            else:
                meta.append(
                    {
                        "name": index,
                        "type": CONTINUOUS,
                        "min": column.min(),
                        "max": column.max(),
                    }
                )
        return meta
    @staticmethod
    def get_metadata_refit(data, meta_file, label_encoder, categorical_columns=tuple(), ordinal_columns=tuple()):
        meta = []

        df = pd.DataFrame(data)
        label_encoder_cursor = 0
        # index here is actually the column name
        for index in df:
            column = df[index]

            if index in categorical_columns:
                # mapper = column.value_counts().index.tolist() label_encoder[label_encoder_cursor].transform(self.df[column].astype(str))
                meta.append({"name": index, "type": CATEGORICAL, "size": len(meta_file['columns'][index]['i2s']), "i2s": label_encoder[label_encoder_cursor].transform(meta_file['columns'][index]['i2s']).tolist()})
                label_encoder_cursor += 1

            elif index in ordinal_columns:
                value_count = list(dict(column.value_counts()).items())
                value_count = sorted(value_count, key=lambda x: -x[1])
                mapper = list(map(lambda x: x[0], value_count))
                meta.append({"name": index, "type": ORDINAL, "size": len(mapper), "i2s": mapper})
            # we accept non-negative type, then it will be seen as continuous
            else:
                meta.append(
                    {
                        "name": index,
                        "type": CONTINUOUS,
                        "min": column.min(),
                        "max": column.max(),
                    }
                )
        return meta
    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        raise NotImplementedError

    def transform(self, data):
        raise NotImplementedError

    def inverse_transform(self, data):
        raise NotImplementedError


class DiscretizeTransformer(Transformer):
    """Discretize continuous columns into several bins.

    Attributes:
        meta
        column_index
        discretizer(sklearn.preprocessing.KBinsDiscretizer)

    Transformation result is a int array.

    """

    def __init__(self, n_bins):
        self.n_bins = n_bins
        self.meta = None
        self.column_index = None
        self.discretizer = None

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.meta = self.get_metadata(data, categorical_columns, ordinal_columns)
        self.column_index = [index for index, info in enumerate(self.meta) if info["type"] == CONTINUOUS]

        self.discretizer = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal", strategy="uniform")

        if not self.column_index:
            return

        self.discretizer.fit(data[:, self.column_index])

    def transform(self, data):
        """Transform data discretizing continuous values.

        Args:
            data(pandas.DataFrame)

        Returns:
            numpy.ndarray

        """
        if self.column_index == []:
            return data.astype("int")

        data[:, self.column_index] = self.discretizer.transform(data[:, self.column_index])
        return data.astype("int")

    def inverse_transform(self, data):
        if self.column_index == []:
            return data

        data = data.astype("float32")
        data[:, self.column_index] = self.discretizer.inverse_transform(data[:, self.column_index])
        return data


class GeneralTransformer(Transformer):
    """Continuous and ordinal columns are normalized to [0, 1].
    Discrete columns are converted to a one-hot vector.
    """

    def __init__(self, act="sigmoid"):
        self.act = act
        self.meta = None
        self.output_dim = None

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.meta = self.get_metadata(data, categorical_columns, ordinal_columns)
        self.output_dim = 0
        for info in self.meta:
            if info["type"] in [CONTINUOUS, ORDINAL]:
                self.output_dim += 1
            else:
                self.output_dim += info["size"]

    def transform(self, data):
        data_t = []
        self.output_info = []
        for id_, info in enumerate(self.meta):
            col = data[:, id_]
            if info["type"] == CONTINUOUS:
                col = (col - (info["min"])) / (info["max"] - info["min"])
                if self.act == "tanh":
                    col = col * 2 - 1
                data_t.append(col.reshape([-1, 1]))
                self.output_info.append((1, self.act))

            elif info["type"] == ORDINAL:
                col = col / info["size"]
                if self.act == "tanh":
                    col = col * 2 - 1
                data_t.append(col.reshape([-1, 1]))
                self.output_info.append((1, self.act))

            else:
                col_t = np.zeros([len(data), info["size"]])
                idx = list(map(info["i2s"].index, col))
                col_t[np.arange(len(data)), idx] = 1
                data_t.append(col_t)
                self.output_info.append((info["size"], "softmax"))
        return np.concatenate(data_t, axis=1)

    def inverse_transform(self, data):
        data_t = np.zeros([len(data), len(self.meta)])

        data = data.copy()

        for id_, info in enumerate(self.meta):
            if info["type"] == CONTINUOUS:
                # every time after dealing with 'data', we reset the 'data' as the rest of 'data'
                current = data[:, 0]
                data = data[:, 1:]

                if self.act == "tanh":
                    current = (current + 1) / 2

                current = np.clip(current, 0, 1)
                data_t[:, id_] = current * (info["max"] - info["min"]) + info["min"]

            elif info["type"] == ORDINAL:
                current = data[:, 0]
                data = data[:, 1:]

                if self.act == "tanh":
                    current = (current + 1) / 2

                current = current * info["size"]
                current = np.round(current).clip(0, info["size"] - 1)
                data_t[:, id_] = current
            else:
                current = data[:, : info["size"]]
                data = data[:, info["size"] :]
                idx = np.argmax(current, axis=1)
                data_t[:, id_] = list(map(info["i2s"].__getitem__, idx))

        return data_t


class GMMTransformer(Transformer):
    """
    Continuous columns are modeled with a GMM.
        and then normalized to a scalor [0, 1] and a n_cluster dimensional vector.

    Discrete and ordinal columns are converted to a one-hot vector.
    """

    def __init__(self, n_clusters=5):
        self.meta = None
        self.n_clusters = n_clusters

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.meta = self.get_metadata(data, categorical_columns, ordinal_columns)
        model = []

        self.output_info = []
        self.output_dim = 0
        for id_, info in enumerate(self.meta):
            if info["type"] == CONTINUOUS:
                gm = GaussianMixture(self.n_clusters)
                gm.fit(data[:, id_].reshape([-1, 1]))
                model.append(gm)
                self.output_info += [(1, "tanh"), (self.n_clusters, "softmax")]
                self.output_dim += 1 + self.n_clusters
            else:
                model.append(None)
                self.output_info += [(info["size"], "softmax")]
                self.output_dim += info["size"]

        self.model = model

    def transform(self, data):
        values = []
        for id_, info in enumerate(self.meta):
            current = data[:, id_]
            if info["type"] == CONTINUOUS:
                current = current.reshape([-1, 1])

                means = self.model[id_].means_.reshape((1, self.n_clusters))
                stds = np.sqrt(self.model[id_].covariances_).reshape((1, self.n_clusters))
                features = (current - means) / (2 * stds)

                probs = self.model[id_].predict_proba(current.reshape([-1, 1]))
                argmax = np.argmax(probs, axis=1)
                idx = np.arange((len(features)))
                features = features[idx, argmax].reshape([-1, 1])

                features = np.clip(features, -0.99, 0.99)

                values += [features, probs]
            else:
                col_t = np.zeros([len(data), info["size"]])
                idx = list(map(info["i2s"].index, current))
                col_t[np.arange(len(data)), idx] = 1
                values.append(col_t)

        return np.concatenate(values, axis=1)

    def inverse_transform(self, data, sigmas):
        data_t = np.zeros([len(data), len(self.meta)])

        st = 0
        for id_, info in enumerate(self.meta):
            if info["type"] == CONTINUOUS:
                u = data[:, st]
                v = data[:, st + 1 : st + 1 + self.n_clusters]
                if sigmas is not None:
                    sig = sigmas[st]
                    u = np.random.normal(u, sig)

                u = np.clip(u, -1, 1)
                st += 1 + self.n_clusters
                means = self.model[id_].means_.reshape([-1])
                stds = np.sqrt(self.model[id_].covariances_).reshape([-1])
                p_argmax = np.argmax(v, axis=1)
                std_t = stds[p_argmax]
                mean_t = means[p_argmax]
                tmp = u * 2 * std_t + mean_t
                data_t[:, id_] = tmp

            else:
                current = data[:, st : st + info["size"]]
                st += info["size"]
                idx = np.argmax(current, axis=1)
                data_t[:, id_] = list(map(info["i2s"].__getitem__, idx))

        return data_t




class BGM_CTGAN_Transformer(Transformer):
    """
    Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete and ordinal columns are converted to a one-hot vector.
    """

    def __init__(self, n_clusters=10, eps=0.005):
        """n_cluster is the upper bound of modes."""
        self.meta = None
        self.n_clusters = n_clusters
        self.eps = eps

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        print("Fitting gaussian mixture models to column data...")

        self.meta = self.get_metadata(data, categorical_columns, ordinal_columns)
        model = []

        self.output_info = []
        self.output_dim = 0
        self.components = []
        self.filter_arr = []
        for id_, info in enumerate(self.meta):
            if info["type"] == CONTINUOUS:
                gm = BayesianGaussianMixture(
                    n_components=self.n_clusters,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=0.001,
                    n_init=1,
                )
                gm.fit(data[:, id_].reshape([-1, 1]))
                model.append(gm)
                comp = gm.weights_ > self.eps
                # now, gm contains 10 gaussian,
                # if weights is less than eps, than it's false, in the later compute, we will neglect these Models
                self.components.append(
                    comp
                )  # comp example: [True, True, False, True, True, False, True, True, True, True]
                self.output_info += [(1, "tanh"), (np.sum(comp), "softmax")]
                self.output_dim += 1 + np.sum(comp)
            
            else:
                model.append(None)
                self.components.append(None)
                self.output_info += [(info["size"], "softmax")]
                self.output_dim += info["size"]

        self.model = model

    def refit(self, data, meta_file, label_encoder, categorical_columns=tuple(), ordinal_columns=tuple(), model = list, components = list):
        print("Refitting gaussian mixture models to column data...")

        self.meta = self.get_metadata_refit(data, meta_file, label_encoder, categorical_columns, ordinal_columns)
        self.output_info = []
        self.output_dim = 0
        self.components = components
        self.model = model
        for id_, info in enumerate(self.meta):

            if info["type"] == CONTINUOUS:
                self.output_info += [(1, "tanh"), (np.sum(self.components[id_]), "softmax")]
                self.output_dim += 1 + np.sum(self.components[id_])
            else:

                self.output_info += [(info["size"], "softmax")]
                self.output_dim += info["size"]


    def get_information(self):
        return self.model, self.components, self.meta

    def set_model(self, model, components):
        self.model = model
        self.components = components

    def transform(self, data, ispositive=False, positive_list=[]):
        values = []
        for id_, info in enumerate(self.meta):
            current = data[:, id_]
            if info["type"] == CONTINUOUS:
                current = current.reshape([-1, 1])
                means = self.model[id_].means_.reshape((1, self.n_clusters))
                stds = np.sqrt(self.model[id_].covariances_).reshape((1, self.n_clusters))
                features = np.empty(shape=(len(current), self.n_clusters))
                if ispositive == True:
                    if id_ in positive_list:
                        features = np.abs(current - means) / (4 * stds)
                else:
                    features = (current - means) / (4 * stds)

                probs = self.model[id_].predict_proba(current.reshape([-1, 1]))
                n_opts = sum(self.components[id_])
                features = features[:, self.components[id_]]
                # probs select only the columns where the components is true
                probs = probs[:, self.components[id_]]

                opt_sel = np.zeros(len(data), dtype="int")
                for i in range(len(data)):
                    pp = probs[i] + 1e-6
                    pp = pp / sum(pp)
                    opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)

                idx = np.arange((len(features)))
                features = features[idx, opt_sel].reshape([-1, 1])
       
                features = np.clip(
                    features, -0.99, 0.99
                )  
                probs_onehot = np.zeros_like(probs)
                probs_onehot[np.arange(len(probs)), opt_sel] = 1
                values += [features, probs_onehot]

            else:
                col_t = np.zeros([len(data), info["size"]])
                idx = list(map(info["i2s"].index, current))
                col_t[np.arange(len(data)), idx] = 1
                values.append(col_t)

        return np.concatenate(values, axis=1)

    def inverse_transform(self, data, sigmas):
        data_t = np.zeros([len(data), len(self.meta)])

        st = 0
        for id_, info in enumerate(self.meta):
            if info["type"] == CONTINUOUS:
                u = data[:, st]
                v = data[:, st + 1 : st + 1 + np.sum(self.components[id_])]

                if sigmas is not None:
                    sig = sigmas[st]
                    u = np.random.normal(u, sig)

                u = np.clip(u, -1, 1)
                v_t = np.ones((data.shape[0], self.n_clusters)) * -100
                v_t[
                    :, self.components[id_]
                ] = v  
                v = v_t
                st += 1 + np.sum(self.components[id_])
                means = self.model[id_].means_.reshape([-1])
                stds = np.sqrt(self.model[id_].covariances_).reshape([-1])
                p_argmax = np.argmax(v, axis=1)
                std_t = stds[p_argmax]
                mean_t = means[p_argmax]
                tmp = u * 4 * std_t + mean_t
                data_t[:, id_] = tmp

            else:
                current = data[:, st : st + info["size"]]
                st += info["size"]
                idx = np.argmax(current, axis=1)
                data_t[:, id_] = list(map(info["i2s"].__getitem__, idx))

        return data_t


class BGMTransformer(Transformer):
    """Model continuous columns with a BayesianGMM and normalized to a scalar [0, 1] and a vector.

    Discrete and ordinal columns are converted to a one-hot vector.
    """

    def __init__(self, n_clusters=10, eps=0.005):
        """n_cluster is the upper bound of modes."""
        self.meta = None
        self.n_clusters = n_clusters
        self.eps = eps

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.meta = self.get_metadata(data, categorical_columns, ordinal_columns)
        model = []

        self.output_info = []
        self.output_dim = 0
        self.components = []
        for id_, info in enumerate(self.meta):
            if info["type"] == CONTINUOUS:
                gm = BayesianGaussianMixture(
                    n_components=self.n_clusters,
                    weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=0.001,
                    n_init=1,
                )
                gm.fit(data[:, id_].reshape([-1, 1]))
                model.append(gm)
                comp = gm.weights_ > self.eps
                # now, gm contains 10 gaussian,
                # if weights is less than eps, than it's false, in the later compute, we will neglect these Models
                self.components.append(comp)

                self.output_info += [(1, "tanh"), (np.sum(comp), "softmax")]
                self.output_dim += 1 + np.sum(comp)
            else:
                model.append(None)
                self.components.append(None)
                self.output_info += [(info["size"], "softmax")]
                self.output_dim += info["size"]

        self.model = model

    def transform(self, data, ispositive=False, positive_list=[]):
        values = []
        for id_, info in enumerate(self.meta):
            current = data[:, id_]
            if info["type"] == CONTINUOUS:
                current = current.reshape([-1, 1])
                means = self.model[id_].means_.reshape((1, self.n_clusters))
                stds = np.sqrt(self.model[id_].covariances_).reshape((1, self.n_clusters))
                features = np.empty(shape=(len(current), self.n_clusters))
                if ispositive == True:
                    if id_ in positive_list:
                        features = np.abs(current - means) / (4 * stds)
                else:
                    features = (current - means) / (4 * stds)

                probs = self.model[id_].predict_proba(current.reshape([-1, 1]))
                n_opts = sum(self.components[id_])
                features = features[:, self.components[id_]]
                probs = probs[:, self.components[id_]]

                opt_sel = np.zeros(len(data), dtype="int")
                for i in range(len(data)):
                    pp = probs[i] + 1e-6
                    pp = pp / sum(pp)
                    opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)

                idx = np.arange((len(features)))
                features = features[idx, opt_sel].reshape([-1, 1])
                features = np.clip(
                    features, -0.99, 0.99
                ) 
                probs_onehot = np.zeros_like(probs)
                probs_onehot[np.arange(len(probs)), opt_sel] = 1
                values += [features, probs_onehot]
            else:

                col_t = np.zeros([len(data), info["size"]])
                idx = list(map(info["i2s"].index, current))
                col_t[np.arange(len(data)), idx] = 1
                values.append(col_t)

        return np.concatenate(values, axis=1)

    def inverse_transform(self, data, sigmas):
        data_t = np.zeros([len(data), len(self.meta)])

        st = 0
        for id_, info in enumerate(self.meta):
            if info["type"] == CONTINUOUS:
                u = data[:, st]
                v = data[:, st + 1 : st + 1 + np.sum(self.components[id_])]

                if sigmas is not None:
                    sig = sigmas[st]
                    u = np.random.normal(u, sig)

                u = np.clip(u, -1, 1)
                v_t = np.ones((data.shape[0], self.n_clusters)) * -100
                v_t[:, self.components[id_]] = v
                v = v_t
                st += 1 + np.sum(self.components[id_])
                means = self.model[id_].means_.reshape([-1])
                stds = np.sqrt(self.model[id_].covariances_).reshape([-1])
                p_argmax = np.argmax(v, axis=1)
                std_t = stds[p_argmax]
                mean_t = means[p_argmax]
                tmp = u * 4 * std_t + mean_t
                data_t[:, id_] = tmp

            else:
                current = data[:, st : st + info["size"]]
                st += info["size"]
                idx = np.argmax(current, axis=1)
                data_t[:, id_] = list(map(info["i2s"].__getitem__, idx))

        return data_t


class TableganTransformer(Transformer):
    def __init__(self, side):
        self.height = side

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple()):
        self.meta = self.get_metadata(data, categorical_columns, ordinal_columns)
        self.minn = np.zeros(len(self.meta))
        self.maxx = np.zeros(len(self.meta))
        for i in range(len(self.meta)):
            if self.meta[i]["type"] == CONTINUOUS:
                self.minn[i] = self.meta[i]["min"] - 1e-3
                self.maxx[i] = self.meta[i]["max"] + 1e-3
            else:
                self.minn[i] = -1e-3
                self.maxx[i] = self.meta[i]["size"] - 1 + 1e-3

    def transform(self, data):
        data = data.copy().astype("float32")
        data = (data - self.minn) / (self.maxx - self.minn) * 2 - 1
        if self.height * self.height > len(data[0]):

            padding = np.zeros((len(data), self.height * self.height - len(data[0])))
            data = np.concatenate([data, padding], axis=1)

        return data.reshape(-1, 1, self.height, self.height)

    def inverse_transform(self, data):
        data = data.reshape(-1, self.height * self.height)

        data_t = np.zeros([len(data), len(self.meta)])

        for id_, info in enumerate(self.meta):
            numerator = data[:, id_].reshape([-1]) + 1
            data_t[:, id_] = (numerator / 2) * (self.maxx[id_] - self.minn[id_]) + self.minn[id_]
            if info["type"] in [CATEGORICAL, ORDINAL]:
                data_t[:, id_] = np.round(data_t[:, id_])

        return data_t


def decode_train_data(data, meta_json, LEs, model_name, pr=None, save=True):
    # Open the meta_file for the dataset
    _file = open(
        meta_json,
    )

    # Load the meta_data
    meta_data = json.load(_file)

    date_time = meta_data["date_info"]

    int_cols = meta_data["integer_info"]

    non_neg_cols = meta_data["non_negative_cols"]
    # Get the column names

    cat_cols = []
    column_names = []

    for column in meta_data["columns"]:
        if column["type"] == "categorical":
            cat_cols.append(column["column_name"])

        column_names.append(column["column_name"])

    # Get the train part of the file.
    df = pd.DataFrame(data, columns=column_names)

    # Using DicLEs/label_encoders, get back categorical values.
    for i in range(len(LEs)):
        le = LEs[i]["label_encoder"]
        df[LEs[i]["column_name"]] = df[LEs[i]["column_name"]].astype(int)
        df[LEs[i]["column_name"]] = le.inverse_transform(df[LEs[i]["column_name"]])

    # For non-negative columns, take inverse log -1 and round up to 0 to get original values.
    for i in df:
        if i in non_neg_cols:
            # print("Nonnegative",i)
            if i in int_cols:
                df[i] = df[i].apply(
                    lambda x: int(np.ceil(np.exp(x) - 1)) if (np.exp(x) - 1) < 0 else int(np.exp(x) - 1)
                )
                # Basically for -999999 taking np.exp(-999999)-1 gives -1. Hence the above code transforms it back to "empty"
                df[i] = df[i].apply(lambda x: "empty" if x == -1 else int(x))
            else:
                df[i] = df[i].apply(lambda x: (np.ceil(np.exp(x) - 1)) if (np.exp(x) - 1) < 0 else (np.exp(x) - 1))
                # Basically for -999999 taking np.exp(-999999)-1 gives -1. Hence the above code transforms it back to "empty"
                df[i] = df[i].apply(lambda x: "empty" if x == -1 else (x))

    # Dealing with date transform.
    if bool(date_time):
        df = Date.inverse_transform(df, date_time)

    # Replace empty with ' '
    df = df.replace("empty", " ")

    for i in int_cols:
        if i in cat_cols:
            df[i] = df[i].apply(lambda x: int(float(x)) if x != " " else " ")
        else:
            df[i] = df[i].apply(lambda x: int(x) if x != " " else " ")

    # Standard output directory.
    output_dir = "data/generated/"

    # Store the data in csv format.
    time_stamp = str(datetime.datetime.now().timestamp()).replace(".", "")
    if save:
        df.to_csv(f"{output_dir}{model_name.split('-')[0]}_{time_stamp}.csv", index=False)

    return df.head(pr) if pr is not None else df
