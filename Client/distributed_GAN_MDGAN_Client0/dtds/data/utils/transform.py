import json
import datetime
import numpy as np
import pandas as pd
from dtds.data.utils.date import Date


class Transform(object):

    @staticmethod
    # TODO restructure and clean
    def inverse(data, meta_json, LEs, epochs=None,fname=None):
        # Open the meta_file for the dataset
        _file = open(meta_json,)

        # Load the meta_data
        meta_data = json.load(_file)

        date_time = meta_data["date_info"]

        non_neg_cols = meta_data['non_negative_cols']
        # Get the column names

        cat_cols = []
        column_names = []

        for column in meta_data["columns"]:
            if (column["type"] == "categorical"):
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
            # print("column: ", i)
            if i in non_neg_cols:
                df[i] = df[i].apply(lambda x: (np.ceil(np.exp(x)-1)) if (np.exp(x)-1) < 0 else (np.exp(x)-1))
                # Basically for -999999 taking np.exp(-999999)-1 gives -1. Hence the above code transforms it back to "empty"
                df[i] = df[i].apply(lambda x: 'empty' if x == -1 else (x))

        # Dealing with date transform.
        if bool(date_time):
            df = Date.inverse_transform(df, date_time)

        # Replace empty with ' '
        df = df.replace('empty', ' ')


        # Assign same name as in meta_file.
        name = meta_data["name"]

        # Standard output directory.
        output_dir = "./"

        # # Store the data in csv format.
        time_stamp = str(datetime.datetime.now().timestamp()).replace('.', '')
        # epochs = '_' + str(epochs) if epochs is not None else ''
        # df.to_csv("{}{}{}.csv".format(output_dir, fname, epochs), index=False)

        return df, time_stamp, column_names