import argparse
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def stat_sim_normalize(real_path,fake_path,cat_cols=None):
    
    #Stat_dict contains statistical similarity for each column.
    Stat_dict={}
    
    #Importing the data. First two are used for correlation distance. As we need to label encode the categories. 
    real = pd.read_csv(real_path)
    fake = pd.read_csv(fake_path)
    really = real.copy()
    fakey = fake.copy()

    #Label encode for correlation distance. 
    if cat_cols:
        for x in cat_cols:
            le = preprocessing.LabelEncoder()
            le.fit(real[x])
            real[x]=le.transform(real[x])
            fake[x]=le.transform(fake[x])
    
    #For categorical columns, JSD is computed, for numeric, wasserstein. 
    for column in real.columns:
        if column in cat_cols:

            real_pdf=(really[column].value_counts()/really[column].value_counts().sum())
            fake_pdf=(fakey[column].value_counts()/fakey[column].value_counts().sum())
 
            categories = (really[column].value_counts()/really[column].value_counts().sum()).keys().tolist()
            categories_fake = (fakey[column].value_counts()/fakey[column].value_counts().sum()).keys().tolist()
            #This part of the code makes sure that both lists are in the right order  
            sorted_categories = sorted(categories)
           
            real_pdf_values = [] 
            fake_pdf_values = []

            for i in sorted_categories:
                real_pdf_values.append(real_pdf[i])
                if i in categories_fake:
                    fake_pdf_values.append(fake_pdf[i]) 
                # in case the categories did not present in fake dataset
                else:
                    fake_pdf_values.append(0) 
                    
            zero_cats = set(really[column].value_counts().keys())-set(fakey[column].value_counts().keys())
            for z in zero_cats:
                real_pdf_values.append(real_pdf[z])
                fake_pdf_values.append(0)  
            Stat_dict[column]=(distance.jensenshannon(real_pdf_values,fake_pdf_values, 2.0))   
        else:
            scaler = MinMaxScaler()
            scaler.fit(real[column].values.reshape(-1,1))
            l1 = scaler.transform(real[column].values.reshape(-1,1)).reshape(1,-1)[0]
            l2 = scaler.transform(fake[column].values.reshape(-1,1)).reshape(1,-1)[0]
            Stat_dict[column]= (wasserstein_distance(l1,l2))

    
    #Computing the averages for numeric and categorical columns. 
    cat_avg = []
    num_avg = []        
    for i in Stat_dict:
        if i in cat_cols:
            cat_avg.append(Stat_dict[i])
        else:
            num_avg.append(Stat_dict[i])
                
    cat_avg = np.mean(cat_avg)
    num_avg = np.mean(num_avg)
    
    return cat_avg,num_avg


  


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("-nepoch", help="number of training epoch")
    args = parser.parse_args()


    # Pass the paths to the data. I have just kept it in the same folder as this script. 
    real_p = "data/raw/Intrusion_train.csv"
    fake_p = "Intrusion_result/Intrusion_synthesis_epoch_"
    categorical_columns = [ 'protocol_type', 'service', 'flag', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'class'] 

    fake_ps = [fake_p+str(i)+".csv" for i in range(int(args.nepoch))]
    result_matrix = []
    for i in range(int(args.nepoch)):
        cat_avg, num_avg = stat_sim_normalize(real_p, fake_ps[i], categorical_columns)
        result_matrix.append([i, cat_avg, num_avg])
    result_df = pd.DataFrame(result_matrix,columns=["Epoch_No.","Avg_JSD","Avg_WD"])
    df = pd.read_csv("timestamp_experiment.csv", header = None)
    for index in df.index:
        if index != 0:
            df.loc[index,0] =  df.loc[index,0] +  df.loc[index-1,0]
    result_df['time_stamp'] = df.loc[:,0]


    result_df.to_csv("Intrusion_statistical_similarity_analysis.csv",index=False)