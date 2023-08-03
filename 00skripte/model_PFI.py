# -*- coding: utf-8 -*-
"""
@author: Benedikt Heudorfer, Tanja Liesch
"""


#%% paths and packages

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import itertools
import copy


# paths
pth_cwd = "C:/Users/Benedikt Heudorfer/Documents/BGR_KNN/06a_Paper Globales Modell/"
os.chdir(pth_cwd)
pth_dt_gw = "./01data/gw"
pth_dt_meteo = "./01data/hyras"
pth_out = "./03grafiken"


#%% functions

# ----- sequentialize  --------------------

def make_sequences(data, n_steps_in, n_input):
    #make the data sequential
    #modified after Jason Brownlee and machinelearningmastery.com
    X, Y = list(), list()
    # step over the entire history one time step at a time
    for i in range(len(data)):
        # find the end of this pattern
        end_idx = i + n_steps_in
        # check if we are beyond the dataset
        if end_idx >= len(data):
            break
        # gather input and output parts of the pattern
        seq_x = data[i:end_idx, :n_input]
        seq_y = data[end_idx, n_input:]
        X.append(seq_x)
        Y.append(seq_y)
    return np.array(X), np.array(Y)

def encode_env(data):

    # Define One Hot Encoder for categorical data
    ohenc = OneHotEncoder(handle_unknown='ignore')    
    
    # Encode "hyraum_gr" categories
    x1 = pd.DataFrame(ohenc.fit_transform(data[["hyraum_gr"]]).toarray())
    x1.columns = ["hyraum_gr_" + str(item) for item in np.arange(0,len(x1.columns))]
    x1.index = data.index
    
    # Encode "huek250_ha" categories
    x2 = pd.DataFrame(ohenc.fit_transform(data[["huek250_ha"]]).toarray())
    x2.columns = ["huek250_ha_" + str(item) for item in np.arange(0,len(x2.columns))]
    x2.index = data.index
    
    # Encode "clc_2018" categories
    x3 = pd.DataFrame(ohenc.fit_transform(data[["clc2018"]]).toarray())
    x3.columns = ["clc2018_" + str(item) for item in np.arange(0,len(x3.columns))]
    x3.index = data.index
    
    # Drop unencoded categorical data
    data_dropped = data.drop(["hyraum_gr","huek250_ha","clc2018"], axis = 1)
    
    # Scaling
    scaler_static = StandardScaler()
    scaler_static.fit(data_dropped)
    data_n = scaler_static.transform(data_dropped)
    
    # Add encoded categorical data (diesn't need to be scaled)
    data_n = np.concatenate([data_n,x1], axis = 1)
    data_n = np.concatenate([data_n,x2], axis = 1)
    data_n = np.concatenate([data_n,x3], axis = 1)

    return data_n

def preprocess():
    
    #Initialize containers
    datafull = list()
    scalers = list()
    X_test = list()
    X_test_stat = list()
    Y_test = list()
    
    
    # Define split dates
    date_start_stop = pd.to_datetime("2008-01-01", format = "%Y-%m-%d")
    date_start_test = pd.to_datetime("2012-01-01", format = "%Y-%m-%d")
    
    
    for i in range(len(dt_list_files)):
        
        # merge all input data
        tempdata = copy.deepcopy(dt_list_meteo[i])
        # tempdata = tempdata.drop(["rH"], axis = 1)
        # tempdata = pd.merge(tempdata,dt_timecode, left_index = True, right_index = True) # add time encoding
        tempdata["ID"] = i
        tempdata = pd.merge(tempdata, dt_list_gw[i], on = "Date", how = "inner")
        
        # save original data for score calculations later on 
        datafull.append(tempdata)
        
        # fit scalers (on train+stop data) and transform (on full) data
        scalers.append(StandardScaler().fit(tempdata[(tempdata.index < date_start_test)]))
        tempdata_n = scalers[i].transform(tempdata)
        
        # Split data
        tempdata_n_train = tempdata_n[(tempdata.index < date_start_stop)]
        tempdata_n_stop = tempdata_n[(tempdata.index >= date_start_stop) & (tempdata.index < date_start_test)]
        tempdata_n_test = tempdata_n[(tempdata.index >= date_start_test)] 
        
        # extend stop + Testdata to be able to fill sequence later
        tempdata_n_stop_ext = np.concatenate([tempdata_n_train[-n_steps_in:], tempdata_n_stop], axis=0)
        tempdata_n_test_ext = np.concatenate([tempdata_n_stop[-n_steps_in:], tempdata_n_test], axis=0)
        
        # sequentialize data and add static features
        temp_x, temp_y = make_sequences(np.asarray(tempdata_n_test_ext), n_steps_in, n_input = 4)
        X_test.append(temp_x); Y_test.append(temp_y[:,1])
        X_test_stat.append(np.repeat(dt_static_features_n[dt_static_features.index == dt_list_names[i]], 
                                     len(temp_x), axis = 0))
    del temp_x,temp_y,tempdata,tempdata_n,tempdata_n_stop,tempdata_n_stop_ext
    del tempdata_n_test,tempdata_n_test_ext,tempdata_n_train
    
    # Final merge
    X_test = np.concatenate(X_test)
    X_test_stat = np.concatenate(X_test_stat)
    Y_test = np.concatenate(Y_test)
    datafull = pd.concat(datafull)
    
    return X_test, X_test_stat, Y_test, datafull
    

#%% load time series data

#-----------------------------------

# list groundwater time series data files
dt_list_files = os.listdir(pth_dt_gw)
temp = [i for i,sublist in enumerate(dt_list_files) if '.csv' in sublist]
dt_list_files = [dt_list_files[i] for i in temp]
del temp

# load groundwater time series
dt_list_gw = list()
for i in range(len(dt_list_files)):
    temp = pd.read_csv(pth_dt_gw + "/" + dt_list_files[i], 
                       parse_dates=['Date'], index_col=0, dayfirst = True, decimal = '.', sep=',')
    dt_list_gw.append(temp)
del temp

# get ID names
dt_list_names = [item[:-12] for item in dt_list_files]

#-----------------------------------

# load meteo data
dt_list_meteo = list()
for i in range(len(dt_list_files)):
    temp = pd.read_csv(pth_dt_meteo+"/"+dt_list_names[i]+'_weeklyData_HYRAS.csv',
                       parse_dates=['Date'],index_col=0, dayfirst = True,
                       decimal = '.', sep=',')
    dt_list_meteo.append(temp)
del temp

#-----------------------------------

# subset: startdate before 2000 and enddate after 2015
fltr = pd.DataFrame({"ID": dt_list_names,
                   "startdate": [item.index[0].year for item in dt_list_gw],
                   "enddate": [item.index[-1].year for item in dt_list_gw]})
fltr_cond1 = [fltr["startdate"] > 2000][0].tolist()
fltr_cond2 = [fltr["enddate"] < 2016][0].tolist()
fltr_getout = []
for i in range(len(dt_list_names)):
    fltr_getout.append(fltr_cond1[i] or fltr_cond2[i])
fltr_getout = [not item for item in fltr_getout]
fltr = fltr[fltr_getout]
dt_list_files = [dt_list_files[i] for i in range(len(dt_list_files)) if fltr_getout[i]]
dt_list_gw = [dt_list_gw[i] for i in range(len(dt_list_gw)) if fltr_getout[i]]
dt_list_names = [dt_list_names[i] for i in range(len(dt_list_names)) if fltr_getout[i]]
dt_list_meteo = [dt_list_meteo[i] for i in range(len(dt_list_meteo)) if fltr_getout[i]]
fltr = fltr.reset_index(drop = True)
del fltr, fltr_cond1, fltr_cond2, fltr_getout

# subset: cut off everything after 2015 and before 1951
dt_list_gw = [item[item.index.year <= 2015] for item in dt_list_gw]
dt_list_gw = [item[item.index.year >= 1951] for item in dt_list_gw]

#subset: drop the stations with missing env feat data
droppy = [item not in ["BB_29519030","BW_177-770-1","MV_16450200","SH_10L55005005"] for item in dt_list_names]
dt_list_files = list(itertools.compress(dt_list_files, droppy))
dt_list_gw = list(itertools.compress(dt_list_gw, droppy))
dt_list_meteo = list(itertools.compress(dt_list_meteo, droppy))
dt_list_names = list(itertools.compress(dt_list_names, droppy))
del droppy


#%% HPs

seeds = [27,63,325,846,1689,3967,28956,74516,192366,455262]

# Hyperparameters of model
n_steps_in = 52
HPi_lstm_size = 128
HPi_static_size =  128
HPi_comb_size =  256
HPi_dropout =  0.1
HPi_targetlr =  0.0003
HPi_epochs = 30
HPi_batchsize = 512

n_dynfeat = 4


for jj in range(2,4):
    
    print("jj="+str(jj))
    
    # choose which stat feature mode
    if jj == 0:
        vari = "tsfeat"
        n_statfeat = 9
    if jj == 1:
        vari = "envfeat"
        n_statfeat = 18
    if jj == 2:
        vari = "rndfeat9"
        n_statfeat = 9
    if jj == 3:
        vari = "rndfeat18"
        n_statfeat = 18
    
    #initialization
    pth_models = "./02ergebnisse/"+vari
    results_dyn = list()
    results_stat = list()
    
    for ii in range(len(seeds)):
        print('\nii='+str(ii)+'\n')
        now1 = datetime.now()
        
        for k in range(n_statfeat):
            print('k='+str(k))
    
            #%% load static features
            
            
            if vari == "tsfeat":
                
                # static time series features
                dt_static_features = pd.read_csv('./01data/features/tsfeat.csv', 
                                                 decimal = '.', sep=';', index_col=[0])
                
                # subsetting
                temp = [item in dt_list_names for item in dt_static_features.index]
                dt_static_features = dt_static_features[temp]
                del temp
                
                # Scaling
                scaler_static = StandardScaler()
                scaler_static.fit(dt_static_features)
                dt_static_features_n = scaler_static.transform(dt_static_features)
            
            #-----------------------------------
            
            
            if vari == "envfeat":
                
                # static environmental features
                dt_static_features = pd.read_csv('./01data/features/envfeat.csv', 
                                                 sep=';', index_col=[0])
                
                # subsetting
                droppy = [item in dt_list_names for item in dt_static_features.index]
                dt_static_features = dt_static_features[droppy]
                
                # encoding routine
                dt_static_features_n = encode_env(dt_static_features)
            
            if vari == "rndfeat9":
                
                # static environmental features
                dt_static_features = pd.read_csv('./01data/features/rndfeat9.csv', 
                                                 sep=';', index_col=[0])
                
                # subsetting
                droppy = [item in dt_list_names for item in dt_static_features.index]
                dt_static_features = dt_static_features[droppy]
                
                # Scaling
                scaler_static = StandardScaler()
                scaler_static.fit(dt_static_features)
                dt_static_features_n = scaler_static.transform(dt_static_features)
                
            if vari == "rndfeat18":
                
                # static environmental features
                dt_static_features = pd.read_csv('./01data/features/rndfeat18.csv', 
                                                 sep=';', index_col=[0])
                
                # subsetting
                droppy = [item in dt_list_names for item in dt_static_features.index]
                dt_static_features = dt_static_features[droppy]
                
                # Scaling
                scaler_static = StandardScaler()
                scaler_static.fit(dt_static_features)
                dt_static_features_n = scaler_static.transform(dt_static_features)
            
            
            #%% data preprocessing
            
            X_test, X_test_stat, Y_test, datafull = preprocess()
                
            #%% PFI modelling
            
            # load saved model checkpoint
            loaded_model = tf.keras.models.load_model(pth_models+'/run'+str(ii)+'/model')
            
            #------------------------------------------------------------------------------
            # PFI for dynamic features
            
            if k == 0:
                # baseline model
                sim_base = loaded_model.predict([X_test, X_test_stat]).squeeze()
                mse_base =  np.mean((sim_base-Y_test) ** 2, axis = 0)
                results_dyn.append({'feature':'baseline', 'seed':seeds[ii], 'mse':mse_base})
            
            if k < n_dynfeat:
                # shuffle feature k
                np.random.seed(seeds[ii])
                save_k = copy.deepcopy(X_test[:,:,k])
                np.random.shuffle(X_test[:,:,k])
            
                # compute out-of-fold MSE with feature k shuffled
                sim_k = loaded_model.predict([X_test, X_test_stat]).squeeze()
                mse_k = np.mean((sim_k-Y_test) ** 2, axis = 0)
                results_dyn.append({'feature':datafull.columns[k], 'seed':seeds[ii], 'mse':mse_k})
                X_test[:,:,k] = save_k
                
               
            #------------------------------------------------------------------------------
            # PFI for dynamic features
            
            if k == 0:
                # baseline model
                sim_base = loaded_model.predict([X_test, X_test_stat]).squeeze()
                mse_base =  np.mean((sim_base-Y_test) ** 2, axis = 0)
                results_stat.append({'feature':'baseline', 'seed':seeds[ii], 'mse':mse_base})
            
            if vari == "tsfeat":
                # shuffle feature k
                np.random.seed(seeds[ii])
                save_k = copy.deepcopy(X_test_stat[:,k])
                np.random.shuffle(X_test_stat[:,k])
            
                # compute out-of-fold MSE with feature k shuffled
                sim_k = loaded_model.predict([X_test, X_test_stat]).squeeze()
                mse_k = np.mean((sim_k-Y_test) ** 2, axis = 0)
                results_stat.append({'feature':dt_static_features.columns[k], 'seed':seeds[ii], 'mse':mse_k})
                X_test_stat[:,k] = save_k
                
            if vari == "envfeat":
                # shuffle feature k
                np.random.seed(seeds[ii])
                np.random.shuffle(dt_static_features.iloc[:,k])
                
                # repeat stat encoding routine
                dt_static_features_n = encode_env(dt_static_features)   
                
                # repeat preprocessing routine
                X_test, X_test_stat, Y_test, datafull = preprocess()
            
                # compute out-of-fold MSE with feature k shuffled
                sim_k = loaded_model.predict([X_test, X_test_stat]).squeeze()
                mse_k = np.mean((sim_k-Y_test) ** 2, axis = 0)
                results_stat.append({'feature':dt_static_features.columns[k], 'seed':seeds[ii], 'mse':mse_k})
        
            if vari == "rndfeat9":
                # shuffle feature k
                np.random.seed(seeds[ii])
                save_k = copy.deepcopy(X_test_stat[:,k])
                np.random.shuffle(X_test_stat[:,k])
            
                # compute out-of-fold MSE with feature k shuffled
                sim_k = loaded_model.predict([X_test, X_test_stat]).squeeze()
                mse_k = np.mean((sim_k-Y_test) ** 2, axis = 0)
                results_stat.append({'feature':dt_static_features.columns[k], 'seed':seeds[ii], 'mse':mse_k})
                X_test_stat[:,k] = save_k
            
            if vari == "rndfeat18":
                # shuffle feature k
                np.random.seed(seeds[ii])
                save_k = copy.deepcopy(X_test_stat[:,k])
                np.random.shuffle(X_test_stat[:,k])
            
                # compute out-of-fold MSE with feature k shuffled
                sim_k = loaded_model.predict([X_test, X_test_stat]).squeeze()
                mse_k = np.mean((sim_k-Y_test) ** 2, axis = 0)
                results_stat.append({'feature':dt_static_features.columns[k], 'seed':seeds[ii], 'mse':mse_k})
                X_test_stat[:,k] = save_k
        
        now2 = datetime.now()
        timetaken = round((now2-now1).total_seconds())/60
        print('\nii='+str(ii)+' took '+str(timetaken)+' minutes\n')
    
    pd.DataFrame(results_dyn).to_csv(pth_models+'/PFI_dyn.csv', float_format='%.4f', 
                                     sep = ";", index=False)
    pd.DataFrame(results_stat).to_csv(pth_models+'/PFI_stat.csv', float_format='%.4f', 
                                      sep = ";", index=False)
    
    
    
