import datetime
import json
import logging
import os
from ast import literal_eval

import numpy as np
import pandas as pd
import pickle5 as pickle
from sklearn import preprocessing

from dtds.data.utils.date import Date
from dtds.data.utils.transform import Transform

logger = logging.getLogger(__name__)


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


class FileGenerator(object):
    """ "
    This class creates the following to files needed for the GAN. The
    meta json file, npz file, and the generated sample csv file.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        file_name: str,
        problem_type: str,
        target_col: str,
        categorical_list: list,
        non_negative_columns: list,
        date_columns: dict = {},
        integer_columns: list = [],
        synthesizer_used: str = "CTGANSynthesizer",
        output_dir: str = "dtds/saved_models/",
    ):
        """
        :param dataframe:               Dataframe uploaded by the user with only the nexessary columns.
        :param file_name:               Name of the raw csv file to look for in the directory path.
        :param categorical_list:        List containing columns that are categorical.
        :param problem_type             Classification type for the dataset.
        :param target_col               Name of the column to generate predictions for the given dataset.
        :param non_negative_columns:    Columns that are continuous but strictly positive.
        :param date_columns:            Dictionary containing date column name with it's respective time format.
        :param synthesizer_used:        The synthesizer chosen by the user.
        """
        self.file_name = file_name.strip()
        self.problem_type = problem_type.strip()
        self.target_col = target_col.strip()
        self.categorical_list = categorical_list
        self.non_negative_columns = non_negative_columns
        self.date_columns = date_columns
        self.integer_columns = []
        self.synthesizer_used = synthesizer_used.strip()
        self.output_dir = output_dir.strip()

        time_stamp = str(datetime.datetime.now().timestamp()).replace(".", "")  # Remove dot from the timestamp
        self.output_name = self.file_name + "_" + self.synthesizer_used + "-" + time_stamp  # Output folder name

        self.df = dataframe

        # Dealing with this first to get the integer columns.
        for i in self.df.columns:

            if "int" in str(self.df[~self.df[i].isnull()][i].dtype):
                self.integer_columns.append(i)
            elif "float" in str(self.df[~self.df[i].isnull()][i].dtype):
                if np.array_equal(self.df[~self.df[i].isnull()][i], self.df[~self.df[i].isnull()][i].astype(int)):
                    self.integer_columns.append(i)

        # print(self.integer_columns)

        # Dealing with missing values.
        self.df = self.df.replace(r" ", np.nan)  # Replace blank cells with NaN
        self.df = self.df.fillna("empty")  # Fill blank cells with the empty string

        all_columns = set(self.df.columns)
        irrelevant_missing_columns = set(self.categorical_list).union(set(self.date_columns.keys()))
        # Extracting the columns for which missing value needs to be fixed.
        # Do not deal with it for now
        relevant_missing_columns = list(all_columns - irrelevant_missing_columns)
        for i in relevant_missing_columns:
            if i in self.non_negative_columns:
                # processing it normally as before.
                self.df[i] = np.log(self.df[i].apply(lambda x: x + 1))

        # not touching date for now but it's not difficult to do this, just need to do some tricks with categorical.
        if bool(self.date_columns):

            self.categorical_list.extend(list(date_columns.keys()))

            self.df = Date.convert(self.df, self.date_columns, self.categorical_list)

        super().__init__()

    def write_data_to_disk(self, meta_data: list, ratio=1.0):
        """
        Function that writes the json and npz files to disk.

        :param meta_data:   Data containing meta information about the columns and the dataset.
        :param ratio:       Ratio defning how big the training and testing sets should be.
        """
        print("in write_data_to_disk")
        # print(meta_data)
        # Prepare path for the output file
        self.output_path = f"{self.output_dir}/{self.output_name}/"

        # Create the dataset directory with the given permission
        os.makedirs(self.output_path, mode=0o760, exist_ok=True)

        # Write json to disk in the newly created dataset directory
        with open("{}{}.json".format(self.output_path, self.output_name), "w") as f:
            json.dump(meta_data, f, sort_keys=True, indent=4, separators=(",", ": "), cls=NumpyEncoder)

    def generate_data(self, meta_file, label_encoder, timestamp):
        """
        Function to generate the meta data for the columns and the dataset.

        :return:    List of LabelEncoders used per categorical column.
        """
        self.output_name = self.file_name + "_" + self.synthesizer_used + "-" + timestamp
        label_encoder_cursor = 0
        for column_index, column in enumerate(self.df.columns):
            if column in self.categorical_list:
                transformed_column = label_encoder[label_encoder_cursor].transform(self.df[column].astype(str))
                self.df[column] = transformed_column
                label_encoder_cursor += 1

        # Prepare path for the output file
        self.output_path = 'dtds/saved_models/{}/'.format(self.output_name)

        # Create the dataset directory with the given permission
        os.makedirs(self.output_path, mode=0o760, exist_ok=True)

        # Write json to disk in the newly created dataset directory
        with open('{}{}.json'.format(self.output_path, self.output_name), 'w') as f:
            json.dump(meta_file, f, sort_keys=True, indent=4, separators=(',', ': '), cls=NumpyEncoder)

        # Split dataset according the the specified ratio
        ratio = 1 # no testing in the npz for now
        margin = int(len(self.df)*ratio)
        train = self.df.values[0:margin]
        test = self.df.values[margin:len(self.df)]

        # Write values of dataframe to an npz file on disk
        np.savez("{}/{}.npz".format(self.output_path, self.output_name), train=train, test=test)
        self.df.to_csv("{}/{}.csv".format(self.output_path, self.output_name), index=False)
        # return meta_data

    def generate_meta_data(self):
        """
        Function to generate the meta data for the columns and the dataset.

        :return:    List of LabelEncoders used per categorical column.
        """
        meta_of_columns = list()
        label_encoders_data = list()

        for column_index, column in enumerate(self.df.columns):
            meta_of_column = dict()
            meta_of_column["column_name"] = column
            if column in self.categorical_list:

                # Add meta data about the column
                meta_of_column["type"] = "categorical"
                meta_of_column["size"] = len(set(self.df[column].astype(str)))
                meta_of_column["i2s"] = dict(self.df[column].astype(str).value_counts())
                # set(self.df[column].astype(str))

            else:
                meta_of_column["type"] = "continous"
                meta_of_column["min"] = np.min(self.df[column])
                meta_of_column["max"] = np.max(self.df[column])

            meta_of_column["column no"] = column_index
            meta_of_columns.append(meta_of_column)

        meta_data = {
            "columns": meta_of_columns,
            "problem_type": self.problem_type,
            "name": self.output_name,
            "date_info": self.date_columns,
            "integer_info": self.integer_columns,
            "non_negative_cols": self.non_negative_columns,
        }

        if self.target_col != "":
            meta_data["target"] = self.target_col
        # print("meta: ", meta_data)
        return meta_data
        # self.write_data_to_disk(meta_data)
        #
        # return label_encoders_data

    def generate_sample_csv_data(self, generated_sample: list):
        """
        Function to save the generated sample to disk in csv format.

        :param generated_sample:    Sample matrix of mxn size.
        """
        with open(
            "dtds/saved_models/model_{}/label_encoders_{}.pickle".format(self.output_name, self.output_name), "rb"
        ) as f:
            label_encoders = pickle.load(f)

        _ = Transform.inverse(generated_sample, "{}{}.json".format(self.output_path, self.output_name), label_encoders)

    def save_synthesizer_model_and_label_encoders(self, synthesizer, label_encoders: dict):
        """
        Function to save current synthesizer to disk in pickle format for faster quering and
        to save the current label encoders for the given dataset to disk in pickle format.

        :param synthesizer:     Current synthesizer model used to generate the sample.
        :param label_encoders:  Encoded values for the labels of the dataset.
        """
        # Create the pickle directory with the given permission
        output_pickle_folder = "dtds/saved_models/{}".format(self.output_name)
        os.makedirs(output_pickle_folder, mode=0o760, exist_ok=True)

        with open("{}/{}.pickle".format(output_pickle_folder, self.output_name), "wb") as f:
            pickle.dump(synthesizer, f, -1)

        with open("{}/label_encoders_{}.pickle".format(output_pickle_folder, self.output_name), "wb") as f:
            pickle.dump(label_encoders, f, protocol=pickle.HIGHEST_PROTOCOL)
