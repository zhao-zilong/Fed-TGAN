import time

import pandas as pd


class Date:
    """ "
    This class contains functionalities that are used for processing date columns.
    It converts the columns that are dates to date objects and splits them into their
    corresponding columns.
    """

    @staticmethod
    def convert(df: pd.DataFrame, date_columns: dict, categorical_list: list):
        """
        Function to convert a single (or multiple) date column(s) to it's respective date type.

        :param df:                  Dataframe of the selected csv file.
        :param date_columns:        List of columns that are dates with their encoding type.
        :param categorical_list:    List containing all of the column names that are categorical.
        :return:                    Dataframe containing the newly added columns without the original date column
                                    List of columns that are dates with their encoding type.
        """
        # Transform it to a date object, Pandas detects and handles it properly
        for column_name in date_columns:
            # TODO maybe convert this to a Set will be faster
            # Remove this column from the categorical list
            categorical_list.remove(column_name)

            format_ = date_columns[column_name].split("|")  # Get the encoded format

            d_format = None
            o_format = None

            if len(format_) == 2:
                d_format = format_[1]
                o_format = format_[0]
            else:
                d_format = format_

            if o_format == "yymmdd":
                # This is done because when there are missing values, the date column becomes casted as float which is not good.
                # df[column_name] = (df[column_name].astype(int)).astype(str)  # Enforce int first then string type on column.
                df[column_name] = df[column_name].apply(
                    lambda x: pd.to_datetime(str(int(x))) if x != "empty" else "empty"
                )  # Auto-convert
            else:
                df[column_name] = df[column_name].astype(str)  # Enforce string type on column
                df[column_name] = df[column_name].apply(
                    lambda x: pd.to_datetime(x) if x != "empty" else "empty"
                )  # Auto-convert

            non_empty_list = []

            for i in range(len(df[column_name])):
                if df[column_name][i] != "empty":
                    non_empty_list.append(i)

            # Split into various columns (the number of columns is dependent on the time encoding)
            df = Date.split(df, column_name, d_format, categorical_list, non_empty_list)

        return df

    @staticmethod
    def split(df: pd.DataFrame, column_name: str, format_: str, categorical_list: list, non_empty_list: list):
        """
        Function to split up a single date column into multiple column parts. The
        original column is removed from the dataframe.

        :param df:                  Dataframe of the selected csv file.
        :param column_column:       Column name of the date column.
        :param format_:             Format of the encoded timestamp.
        :param categorical_list:    List containing all of the column names that are categorical.
        :param non_empty_list:      List containing all the non-empty indexes.
        :return:                    Dataframe containing the newly added columns without the original
                                    date column.
        """
        encode = {"YYYY": "-year", "MM": "-month", "DD": "-day", "hh": "-hour", "mm": "-minute", "ss": "-second"}

        format_elements = format_.split("-")
        for format_element in format_elements:
            if format_element == "YYYY":
                df[column_name + encode[format_element]] = "empty"
                df.loc[non_empty_list, column_name + encode[format_element]] = df[column_name][
                    non_empty_list
                ].dt.strftime("%y")
            elif format_element == "MM":
                df[column_name + encode[format_element]] = "empty"
                df.loc[non_empty_list, column_name + encode[format_element]] = df[column_name][
                    non_empty_list
                ].dt.strftime("%m")
            elif format_element == "DD":
                df[column_name + encode[format_element]] = "empty"
                df.loc[non_empty_list, column_name + encode[format_element]] = df[column_name][
                    non_empty_list
                ].dt.strftime("%d")
            elif format_element == "hh":
                df[column_name + encode[format_element]] = "empty"
                df.loc[non_empty_list, column_name + encode[format_element]] = df[column_name][
                    non_empty_list
                ].dt.strftime("%H")
            elif format_element == "mm":
                df[column_name + encode[format_element]] = "empty"
                df.loc[non_empty_list, column_name + encode[format_element]] = df[column_name][
                    non_empty_list
                ].dt.strftime("%M")
            else:  # ss
                df[column_name + encode[format_element]] = "empty"
                df.loc[non_empty_list, column_name + encode[format_element]] = df[column_name][
                    non_empty_list
                ].dt.strftime("%S")

            categorical_list.append(column_name + encode[format_element])

        # Remove original date column from dataFrame
        df = df.drop([column_name], axis=1)
        return df

    @staticmethod
    def inverse_transform(df: pd.DataFrame, date_columns: dict):
        """
        Function to convert a single (or multiple) date column(s) to back to it's original respective date type.

        :param df:                  Dataframe of the selected csv file.
        :param date_columns:        List of columns that are dates with their encoding type.
        :return:                    Dataframe containing the original date columns

        """
        encode = {"YYYY": "-year", "MM": "-month", "DD": "-day", "hh": "-hour", "mm": "-minute", "ss": "-second"}

        # Transform it to a date object, Pandas detects and handles it properly
        for column_name in date_columns:

            format_ = date_columns[column_name].split("|")  # Get the encoded format
            o_format = None
            d_format = None
            if len(format_) == 2:
                o_format = format_[0]
                d_format = format_[1]
            else:
                d_format = format_
            # dformat = format.split('|')[0].strip()
            # cformat = format.split('|')[1].strip()
            format_elements = d_format.split("-")
            concat = []
            for format_element in format_elements:
                concat.append(column_name + encode[format_element])
            for i in concat:
                df[i] = df[i].astype(str)

            df[column_name] = df[concat].apply(lambda x: "-".join(x), axis=1)

            # valid date range
            vdr = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
            k = 0
            for i in df[column_name]:
                if "empty" in i:
                    df.loc[k, column_name] = "empty"
                else:
                    split_i = i.split("-")
                    if int(split_i[2]) > int(vdr[int(split_i[1])]):
                        # print("Invalid",i)
                        if split_i[1] == "02":  # | (split_i[1]=="2"):# :
                            # print("Do you ever even come here?")
                            # print("Invalid",i)
                            if (
                                ((int(split_i[0]) % 4) == 0)
                                & ((int(split_i[0]) % 100) == 0)
                                & ((int(split_i[0]) % 400) == 0)
                            ):
                                split_i[2] = "29"
                            else:
                                split_i[2] = "28"
                        else:
                            split_i[2] = "30"
                    nd = "-".join(split_i)
                    print(nd)
                    df.loc[k, column_name] = nd
                k += 1

            # pattern=''
            # if len(concat)==3:
            #    pattern = '%Y-%M-%d' # %H:%M:%S for later use.

            # if dformat == "Epoch time":
            # Temp fix only for berka, this whole epoch time was not really epoch time it was yymmdd.
            # df[column_name] = df[column_name].apply(lambda x: (time.mktime(time.strptime(x, pattern))))

            if o_format == None:
                df[column_name] = df[column_name].apply(lambda x: pd.to_datetime(x) if x != "empty" else "empty")
            elif o_format == "yymmdd":
                df[column_name] = df[column_name].apply(lambda x: pd.to_datetime(x) if x != "empty" else "empty")
                df[column_name] = df[column_name].apply(
                    lambda x: int("".join([x.strftime("%y"), x.strftime("%m"), x.strftime("%d")]))
                    if x != "empty"
                    else "empty"
                )

        df = df.drop(columns=concat)
        return df


# For testing that it works, ofcourse none of this code has front end or back end support with respect to training yet.

# if __name__ == "__main__":

#    df = pd.read_csv('D:\\Learning_Material\\Research_Assistant\\aegon_project-master\\berka_s.csv')
#    df = Date.convert(df, {"date": "yymmdd|YYYY-MM-DD"},["date"])
#    df = pd.read_csv("D:\\Learning_Material\\Research_Assistant\\aegon_project-master\\aegon\\ctgan\\generated_csv_files\\berka_120_raw_CTGANSynthesizer1603471421889298.csv")
#    print(df)
#    df = Date.inverse_transform(df, {"date": "yymmdd|YYYY-MM-DD"})
#    print(df)
