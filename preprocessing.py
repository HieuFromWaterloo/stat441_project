import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import warnings
warnings.filterwarnings(action="ignore")

# define one hot encoding for categorical var


def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    categorical_columns.remove("deposit")
    # apply drop_first to deal with multicollinearity
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category, drop_first=True)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Memory reduction: some members do not have sufficient compuation capacity


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def transform_pipeline(data):
    df = data.copy()
    # log transform balance
    df["balance"] = np.log(df["balance"] + abs(df["balance"].min()) + 1)
    # transformation in categorical features
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    for col in categorical_columns:
        if col not in ["contact", "poutcome"]:
            """
            besides contact and poutcome variables, "unknown" values will be replaced
            by the mode in each categorical column
            """
            data[col].replace('unknown', data[col].mode()[0], inplace=True)
    # replace -1 by 999
    df["pdays"].replace(-1, 999, inplace=True)
    # regroup the months into 3 main groups: early, mid, end of each quarter in a year
    df["month"] = df["month"].apply(lambda x: "EarlyQuarter" if (x in ["jan", "apr", "jul", "oct"])
                                    else ("MidQuarter" if (x in ["feb", "may", "aug", "oct"]) else
                                          "EndQuarter"))
    # regroup the day into 3 main groups based on the density of y/n ratios
    df["day"] = df["day"].apply(lambda x: "gr1" if (x in [1, 10])
                                else ("gr2" if (x in [19, 20, 28, 29, 31]) else "gr3"))
    # replace other with unknown
    df["poutcome"].replace("other", "unknown", inplace=True)

    # apply one hot encoding on categorical variables
    df, _ = one_hot_encoder(df)

    # get rid of some of the null columns
    for col in ["month_nan", "contact_nan", "loan_nan", "housing_nan", "default_nan",
                "education_nan", "marital_nan", "job_nan", "day_nan", "poutcome_nan"]:
        del df[col]
    # memory optimization before outputing the final dataframe
    df = reduce_mem_usage(df)
    return df


def normalization(traindf, df):
    for col in ["age", "balance", "duration", "campaign", "pdays", "previous"]:
        if col != "balance":
            df[col] = (df[col] - traindf[col].mean()) / traindf[col].std()
        else:
            # for some reasons trainset["balance"].mean() results into "inf" after the log transformation so we hard coded the train mean here
            train_mean = 8.995432292407646
            df[col] = (df[col] - train_mean) / traindf[col].std()
    return df


def split_data(df):
    # Shuffle dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    # train test split: 70/30 ratio from the entire data set
    trainset, testset = train_test_split(df, test_size=0.3, random_state=44119)
    # split valid and test: 50/50 ratio from the test set
    validset, testset = train_test_split(testset, test_size=0.5, random_state=44119)

    # normalization
    trainset = normalization(trainset, trainset)
    validset = normalization(trainset, validset)
    testset = normalization(trainset, testset)
    return trainset, validset, testset


def main():
    df = pd.read_csv("data/bank.csv", sep=",")
    df = transform_pipeline(df)
    train, valid, test = split_data(df)
