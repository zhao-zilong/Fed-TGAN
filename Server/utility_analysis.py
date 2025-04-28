import argparse
import pandas as pd
import numpy as np
from sklearn import preprocessing, tree, ensemble,linear_model, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def real_res(original_real, real, test, target_col,cat_cols):
    """
    original_real: It is the original data which the train and the test dataset are splitted from it.
    real: The real train dataset
    test: The real test dataset
    target_col: target column of the dataset, this will be used as the label of the data
    cat_cols: categorical columns name

    return: List of metrics (i.e., accuracy and f1 score) from logistic regression, decision tree, random forest, and MLP.
    """
    real = real.copy()
    test = test.copy()
    original_real = original_real.copy()

    target_col = target_col   
    cat_cols = cat_cols

    if cat_cols:
        for x in cat_cols:
            le = preprocessing.LabelEncoder()
            real[x] = real[x].astype(str)
            test[x] = test[x].astype(str)
            original_real[x] = original_real[x].astype(str)
            le.fit(original_real[x].values)
            real[x]=le.transform(real[x])
            test[x]=le.transform(test[x])
            original_real[x]=le.transform(original_real[x])
            
    y_train_real = real[target_col]
    X_train_real = real.drop(columns=[target_col])
    y_test_real = test[target_col]
    X_test_real = test.drop(columns=[target_col])
    original_real = original_real.drop(columns=[target_col])
        
    scaler_real = preprocessing.StandardScaler().fit(original_real.values)
    X_train_scaled_real = scaler_real.transform(X_train_real)
    X_test_scaled_real =  scaler_real.transform(X_test_real)
                                
    real_utility = []

    print("training of LR")
    lr_real = linear_model.LogisticRegression(class_weight="balanced",random_state=69).fit(X_train_scaled_real,y_train_real)
    y_lr_real = lr_real.predict(X_test_scaled_real)
    y_proba_lr_real =  lr_real.predict_proba(X_test_scaled_real)

    real_utility.append([metrics.accuracy_score(y_test_real,y_lr_real),
    f1_score(y_test_real,y_lr_real, average='weighted')
                        ])

    print("training of DT")
    dt_real = tree.DecisionTreeClassifier(class_weight="balanced",random_state=69).fit(X_train_scaled_real,y_train_real)
    y_dt_real = dt_real.predict(X_test_scaled_real)
    y_proba_dt_real =  dt_real.predict_proba(X_test_scaled_real)

    real_utility.append([metrics.accuracy_score(y_test_real,y_dt_real),
    f1_score(y_test_real,y_dt_real, average='weighted')
                        ])

    print("training of RF")
    rf_real = ensemble.RandomForestClassifier(class_weight="balanced",random_state=69).fit(X_train_scaled_real,y_train_real)
    y_rf_real = rf_real.predict(X_test_scaled_real)
    y_proba_rf_real =  rf_real.predict_proba(X_test_scaled_real)

    real_utility.append([metrics.accuracy_score(y_test_real,y_rf_real),
    f1_score(y_test_real,y_rf_real, average='weighted')
                        ])

    print("training of MLP")
    ml_real = MLPClassifier(random_state=69).fit(X_train_scaled_real,y_train_real)
    y_ml_real = ml_real.predict(X_test_scaled_real)
    y_proba_ml_real =  ml_real.predict_proba(X_test_scaled_real)

    real_utility.append([metrics.accuracy_score(y_test_real,y_ml_real),
    f1_score(y_test_real,y_ml_real, average='weighted')
                        ])

    return real_utility


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-train_path", default = 'data/raw/Intrusion_train.csv', help="path to train dataset")
    parser.add_argument("-test_path", default = 'data/raw/Intrusion_test.csv', help="path to test dataset")
    parser.add_argument("-synthetic_path", default = 'Intrusion_result/Intrusion_synthesis_epoch_0.csv', help="path to synthetic dataset")
    args = parser.parse_args()

    real = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    fake = pd.read_csv(args.synthetic_path)
    # Intrusion
    cat_cols = [ 'protocol_type', 'service', 'flag', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'class']
    target_col = 'class'
    original_real = pd.concat([real, test])
    print("=========== evaluation for real data===============")
    real_utility = real_res(original_real, real, test, target_col, cat_cols)
    print("=========== evaluation for synthetic data===============")
    fake_utility = real_res(original_real, fake, test, target_col, cat_cols)
    diff_utility = np.array(real_utility) - np.array(fake_utility)
    print("difference in accuracy and f1-score for all AL algorithms: ", diff_utility)
    print("difference in f1-score: ", diff_utility.mean(axis=0)[1])