# -*- coding: utf-8 -*-
"""
@author: Benedikt Heudorfer, Tanja Liesch
"""

# select model variant and let the code run
modelvars = ["onlydyn_lstm","envfeat_lstm","tsfeat_lstm","envfeat_cnn","tsfeat_cnn",
             "rndfeat9_lstm","rndfeat18_lstm"]
modelvar = modelvars[0]


#%% paths and packages

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import itertools
import copy

# paths
pth_cwd = "C:/Users/Benedikt Heudorfer/Documents/BGR_KNN/06a_Paper Globales Modell/"
os.chdir(pth_cwd)
pth_dt_gw = "./01data/gw"
pth_dt_meteo = "./01data/hyras"

if modelvar == "onlydyn_lstm":
    pth_out = "./02ergebnisse/ablation_onlydyn"   
if modelvar == "envfeat_lstm":
    pth_out = "./02ergebnisse/envfeat"   
if modelvar == "tsfeat_lstm":
    pth_out = "./02ergebnisse/tsfeat"
if modelvar == "envfeat_cnn":
    pth_out = "./02ergebnisse/envfeat_CNN"
if modelvar == "tsfeat_cnn":
    pth_out = "./02ergebnisse/tsfeat_CNN"
if modelvar == "rndfeat9_lstm":
    pth_out = "./02ergebnisse/rndfeat9"
if modelvar == "rndfeat18_lstm":
    pth_out = "./02ergebnisse/rndfeat18"


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

# ----- learning rate scheduling  --------------------

# from: https://stackabuse.com/learning-rate-warmup-with-cosine-decay-in-keras-and-tensorflow/

# Define warmup & learning rate decay procedure
def lr_warmup_cosine_decay(global_step,
                           warmup_steps,
                           hold = 0,
                           total_steps=0,
                           start_lr=0.0,
                           target_lr=0.001):
    # Cosine decay
    learning_rate = 0.5 * target_lr * (1 + np.cos(np.pi * (global_step - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))

    # Target LR * progress of warmup (=1 at the final warmup step)
    warmup_lr = target_lr * (global_step / warmup_steps)

    # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and hold
    if hold > 0:
        learning_rate = np.where(global_step > warmup_steps + hold, learning_rate, target_lr)
    
    learning_rate = np.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate

# Define warmup & learning rate schedule as a callback 
class WarmupCosineDecay(tf.keras.callbacks.Callback):
    def __init__(self, total_steps=0, warmup_steps=0, start_lr=0.0, target_lr=1e-3, hold=0):

        super(WarmupCosineDecay, self).__init__()
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.global_step = 0
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.lrs = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = model.optimizer.lr.numpy()
        self.lrs.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = lr_warmup_cosine_decay(global_step=self.global_step,
                                    total_steps=self.total_steps,
                                    warmup_steps=self.warmup_steps,
                                    start_lr=self.start_lr,
                                    target_lr=self.target_lr,
                                    hold=self.hold)
        K.set_value(self.model.optimizer.lr, lr)


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
# a little dirty coding but it works
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

# save IDs of remaining stations
pd.DataFrame({"ID": dt_list_names}).to_csv("./01data/IDremainig.csv", sep = ";")


#%% load static features

if modelvar in ["tsfeat_lstm","tsfeat_cnn","onlydyn_lstm","rndfeat9_lstm","rndfeat18_lstm"]: 
    
    # static time series features
    dt_static_features = pd.read_csv('./01data/features/tsfeat.csv', 
                                     decimal = '.', sep=';', index_col=[0])
    if modelvar == "rndfeat9_lstm":
        dt_static_features = pd.read_csv('./01data/features/rndfeat9.csv', 
                                         decimal = '.', sep=';', index_col=[0])
    if modelvar == "rndfeat18_lstm":
        dt_static_features = pd.read_csv('./01data/features/rndfeat18.csv', 
                                         decimal = '.', sep=';', index_col=[0])
        
    # subsetting
    temp = [item in dt_list_names for item in dt_static_features.index]
    dt_static_features = dt_static_features[temp]
    del temp
    
    # Scaling
    scaler_static = StandardScaler()
    scaler_static.fit(dt_static_features)
    dt_static_features_n = scaler_static.transform(dt_static_features)



if modelvar in ["envfeat_lstm","envfeat_cnn"]:
    
    # static environmental features
    dt_static_features = pd.read_csv('./01data/features/envfeat.csv', sep=';', index_col=[0])
    
    # subsetting
    droppy = [item in dt_list_names for item in dt_static_features.index]
    dt_static_features = dt_static_features[droppy]
    
    # Define One Hot Encoder for categorical data
    ohenc = OneHotEncoder(handle_unknown='ignore')
    
    # Encode "hyraum_gr" categories
    x1 = pd.DataFrame(ohenc.fit_transform(dt_static_features[["hyraum_gr"]]).toarray())
    x1.columns = ["hyraum_gr_" + str(item) for item in np.arange(0,len(x1.columns))]
    x1.index = dt_static_features.index
    
    # Encode "huek250_ha" categories
    x2 = pd.DataFrame(ohenc.fit_transform(dt_static_features[["huek250_ha"]]).toarray())
    x2.columns = ["huek250_ha_" + str(item) for item in np.arange(0,len(x2.columns))]
    x2.index = dt_static_features.index
    
    # Encode "clc_2018" categories
    x3 = pd.DataFrame(ohenc.fit_transform(dt_static_features[["clc2018"]]).toarray())
    x3.columns = ["clc2018_" + str(item) for item in np.arange(0,len(x3.columns))]
    x3.index = dt_static_features.index
    
    # Drop unencoded categorical data
    dt_static_features = dt_static_features.drop(["hyraum_gr","huek250_ha","clc2018"], axis = 1)
    
    # Scaling
    scaler_static = StandardScaler()
    scaler_static.fit(dt_static_features)
    dt_static_features_n = scaler_static.transform(dt_static_features)
    
    # Add encoded categorical data (diesn't need to be scaled)
    dt_static_features_n = np.concatenate([dt_static_features_n,x1], axis = 1)
    dt_static_features_n = np.concatenate([dt_static_features_n,x2], axis = 1)
    dt_static_features_n = np.concatenate([dt_static_features_n,x3], axis = 1)


#%% HPs

HP_seeds = [27,63,325,846,1689,3967,28956,74516,192366,455262]

# Hyperparameters
n_steps_in = 52
HPi_lstm_size = 128
HPi_static_size =  128
HPi_comb_size =  256
HPi_dropout =  0.1
HPi_targetlr =  0.0003
HPi_epochs = 30
HPi_batchsize = 512

# ii=0
for ii in range(len(HP_seeds)):
    HPi_seed = HP_seeds[ii]
    
    # create folder for outputs
    pth_out_i = pth_out+'/run'+str(ii)
    os.mkdir(pth_out_i)
    
    
    #%% data preprocessing
    
    #Initialize containers
    IDlen = list()
    datafull = list()
    scalers = list()
    scalers_y = list()
    X_train,Y_train = list(), list()
    X_stop,Y_stop = list(), list()
    X_test,Y_test = list(), list()
    X_train_stat, X_stop_stat, X_test_stat = list(),list(),list()
    
    # Define split dates
    date_start_stop = pd.to_datetime("2008-01-01", format = "%Y-%m-%d")
    date_start_test = pd.to_datetime("2012-01-01", format = "%Y-%m-%d")
    
    for i in range(len(dt_list_files)):
        
        # merge all input data
        tempdata = copy.deepcopy(dt_list_meteo[i])
        tempdata["ID"] = i
        tempdata = pd.merge(tempdata, dt_list_gw[i], on = "Date", how = "inner")
        
        # save original data for score calculations later on 
        datafull.append(tempdata)
        
        # fit scalers (on train+stop data) and transform (on full) data
        scalers.append(StandardScaler().fit(tempdata[(tempdata.index < date_start_test)]))
        scalers_y.append(StandardScaler().fit(pd.DataFrame(tempdata[(tempdata.index < date_start_test)]["GWL"])))
        tempdata_n = scalers[i].transform(tempdata)
        
        # Split data
        tempdata_n_train = tempdata_n[(tempdata.index < date_start_stop)]
        tempdata_n_stop = tempdata_n[(tempdata.index >= date_start_stop) & (tempdata.index < date_start_test)]
        tempdata_n_test = tempdata_n[(tempdata.index >= date_start_test)] 
        
        # ID tracker: Save length of test set to identify individual Mst after modelfit
        IDlen.append(np.repeat(i,len(tempdata_n_test)))
        
        # extend stop + Testdata to be able to fill sequence later
        tempdata_n_stop_ext = np.concatenate([tempdata_n_train[-n_steps_in:], tempdata_n_stop], axis=0)
        tempdata_n_test_ext = np.concatenate([tempdata_n_stop[-n_steps_in:], tempdata_n_test], axis=0)
        
        # sequentialize data and add static features
        temp_x, temp_y = make_sequences(np.asarray(tempdata_n_train), n_steps_in, n_input = 4)
        X_train.append(temp_x); Y_train.append(temp_y[:,1])
        X_train_stat.append(np.repeat(dt_static_features_n[dt_static_features.index == dt_list_names[i]], 
                                      len(temp_x), axis = 0))
        
        temp_x, temp_y = make_sequences(np.asarray(tempdata_n_stop_ext), n_steps_in, n_input = 4)
        X_stop.append(temp_x); Y_stop.append(temp_y[:,1])
        X_stop_stat.append(np.repeat(dt_static_features_n[dt_static_features.index == dt_list_names[i]], 
                                     len(temp_x), axis = 0))
        
        temp_x, temp_y = make_sequences(np.asarray(tempdata_n_test_ext), n_steps_in, n_input = 4)
        X_test.append(temp_x); Y_test.append(temp_y[:,1])
        X_test_stat.append(np.repeat(dt_static_features_n[dt_static_features.index == dt_list_names[i]], 
                                     len(temp_x), axis = 0))
    del temp_x,temp_y,tempdata,tempdata_n,tempdata_n_stop,tempdata_n_stop_ext
    del tempdata_n_test,tempdata_n_test_ext,tempdata_n_train
    
    # Final merge
    X_train = np.concatenate(X_train)
    X_train_stat = np.concatenate(X_train_stat)
    Y_train = np.concatenate(Y_train)
    X_stop = np.concatenate(X_stop)
    X_stop_stat = np.concatenate(X_stop_stat)
    Y_stop = np.concatenate(Y_stop)
    X_test = np.concatenate(X_test)
    X_test_stat = np.concatenate(X_test_stat)
    Y_test = np.concatenate(Y_test)
    IDlen = np.concatenate(IDlen)
    datafull = pd.concat(datafull)
    datafullwide = pd.pivot(datafull.drop(["P",'rH',"T","Tsin"], axis = 1), columns = "ID")
    
    
    #%% modelling
    
    #-----------------------------    
    # Model
    #-----------------------------
    
    #take time
    now1 = datetime.now()
    
    # set seed
    np.random.seed(HPi_seed)
    tf.random.set_seed(HPi_seed)
    
    # Callbacks
    model_mc = tf.keras.callbacks.ModelCheckpoint(filepath=pth_out_i+'/model', 
                                                  save_best_only=True)
    model_wcd = WarmupCosineDecay(total_steps=len(X_train)/HPi_batchsize*HPi_epochs, 
                                  warmup_steps=int(len(X_train)/HPi_batchsize),
                                  hold=0, 
                                  target_lr=HPi_targetlr)
    
    if modelvar == "onlydyn_lstm":
        # Input layers for dynamic and static model strands
        model_dyn_in = tf.keras.Input(shape=(n_steps_in, X_train.shape[2]))
        # Dynamic model strand
        model_dyn = tf.keras.layers.LSTM(HPi_lstm_size, activation='relu')(model_dyn_in)
        model_dyn = tf.keras.layers.Dropout(HPi_dropout)(model_dyn)
        # Combine dynamic and static strands
        model_comb = tf.keras.layers.Dense(HPi_comb_size, activation='relu')(model_dyn)
        model_comb = tf.keras.layers.Dropout(HPi_dropout)(model_comb)
        # Define output layer for predictions
        model_output = tf.keras.layers.Dense(units=1, activation='linear')(model_comb)
        # Define model with both dynamic and static inputs
        model = tf.keras.Model(inputs=[model_dyn_in], outputs=model_output)
        # Compile model with appropriate loss function and optimizer
        optimizer =  tf.keras.optimizers.Adam(epsilon = 0.0001)
        model.compile(loss='mse', optimizer=optimizer)
        
        # Train model on training data
        model_history = model.fit([X_train], Y_train, 
                                  validation_data=([X_stop], Y_stop),
                                  epochs=HPi_epochs, verbose=2, batch_size=HPi_batchsize, 
                                  shuffle=True, callbacks=[model_mc,model_wcd])
    
    if modelvar in ["tsfeat_lstm","envfeat_lstm"]: 
        # Input layers for dynamic and static model strands
        model_dyn_in = tf.keras.Input(shape=(n_steps_in, X_train.shape[2]))
        model_stat_in = tf.keras.Input(shape=(X_train_stat.shape[1],))
        # Dynamic model strand
        model_dyn = tf.keras.layers.LSTM(HPi_lstm_size, activation='relu')(model_dyn_in)
        model_dyn = tf.keras.layers.Dropout(HPi_dropout)(model_dyn)
        # Static model strand
        model_stat = tf.keras.layers.Dense(HPi_static_size, activation='relu')(model_stat_in)
        model_stat = tf.keras.layers.Dropout(HPi_dropout)(model_stat)
        # Combine dynamic and static strands
        model_comb = tf.keras.layers.concatenate([model_dyn, model_stat])
        model_comb = tf.keras.layers.Dense(HPi_comb_size, activation='relu')(model_comb)
        model_comb = tf.keras.layers.Dropout(HPi_dropout)(model_comb)
        # Define output layer for predictions
        model_output = tf.keras.layers.Dense(units=1, activation='linear')(model_comb)
        # Define model with both dynamic and static inputs
        model = tf.keras.Model(inputs=[model_dyn_in, model_stat_in], outputs=model_output)
        # Compile model with appropriate loss function and optimizer
        optimizer =  tf.keras.optimizers.Adam(epsilon = 0.0001)
        model.compile(loss='mse', optimizer=optimizer)
    
        # Train model on training data
        model_history = model.fit([X_train, X_train_stat], Y_train, 
                                  validation_data=([X_stop, X_stop_stat], Y_stop),
                                  epochs=HPi_epochs, verbose=2, batch_size=HPi_batchsize, 
                                  shuffle=True, callbacks=[model_mc,model_wcd])
        
    if modelvar in ["tsfeat_cnn","envfeat_cnn"]: 
        # Input layers for dynamic and static model strands
        model_dyn_in = tf.keras.Input(shape=(n_steps_in, X_train.shape[2]))
        model_stat_in = tf.keras.Input(shape=(X_train_stat.shape[1],))
        # Dynamic model strand
        model_dyn = tf.keras.layers.Conv1D(filters=HPi_lstm_size,
                                           kernel_size=3,
                                           activation='relu',
                                           padding='same')(model_dyn_in)
        model_dyn = tf.keras.layers.BatchNormalization()(model_dyn)
        model_dyn = tf.keras.layers.MaxPool1D(padding='same')(model_dyn)
        model_dyn = tf.keras.layers.Dropout(HPi_dropout)(model_dyn)
        model_dyn = tf.keras.layers.Flatten()(model_dyn)
        # Static model strand
        model_stat = tf.keras.layers.Dense(HPi_static_size, activation='relu')(model_stat_in)
        model_stat = tf.keras.layers.Dropout(HPi_dropout)(model_stat)
        # Combine dynamic and static strands
        model_comb = tf.keras.layers.concatenate([model_dyn, model_stat])
        model_comb = tf.keras.layers.Dense(HPi_comb_size, activation='relu')(model_comb)
        model_comb = tf.keras.layers.Dropout(HPi_dropout)(model_comb)
        # Define output layer for predictions
        model_output = tf.keras.layers.Dense(units=1, activation='linear')(model_comb)
        # Define model with both dynamic and static inputs
        model = tf.keras.Model(inputs=[model_dyn_in, model_stat_in], outputs=model_output)
        # Compile model with appropriate loss function and optimizer
        optimizer =  tf.keras.optimizers.Adam(epsilon = 0.0001)
        model.compile(loss='mse', optimizer=optimizer)
        
        model_history = model.fit([X_train, X_train_stat], Y_train, 
                                  validation_data=([X_stop, X_stop_stat], Y_stop),
                                  epochs=HPi_epochs, verbose=2, batch_size=HPi_batchsize, 
                                  shuffle=True, callbacks=[model_mc,model_wcd])
        
    # take time
    now2 = datetime.now()
    timetaken = round((now2-now1).total_seconds())/60
    print('\n timetaken = '+str(timetaken)+'\n')
    
    if modelvar == "onlydyn_lstm":
        # predict - with saved model checkpoint
        loaded_model = tf.keras.models.load_model(pth_out_i+'/model')
        sim_n = loaded_model.predict([X_test])
    
    if modelvar in ["tsfeat_lstm","envfeat_lstm","tsfeat_cnn","envfeat_cnn"]:     
        # predict - with saved model checkpoint
        loaded_model = tf.keras.models.load_model(pth_out_i+'/model')
        sim_n = loaded_model.predict([X_test, X_test_stat])
    
    # inverse scaling
    sim = []
    for i in range(len(dt_list_names)):
        temp = sim_n[IDlen == i,0]
        temp = scalers_y[i].inverse_transform(temp)
        temp = pd.DataFrame({"ID": np.repeat(i,len(temp)), "sim": temp})
        sim.append(temp)
    del temp
    sim = pd.concat(sim)
    
    
    
    #%% Evaluate
    
    # Minimum val MSE
    MSEvalmin = np.min(model_history.history['val_loss'])
    
    results = []
    for i in range(len(dt_list_names)):
        temp = sim[sim.ID == i]
        temp.columns = (dt_list_names[i] + "_" + temp.columns).tolist()
        temp[dt_list_names[i]+"_obs"] = scalers_y[i].inverse_transform(Y_test[IDlen == i])
        temp = temp.reset_index(drop = True)
        results.append(temp)
    del temp
    results = pd.concat(results, axis = 1)
    
    sim_test = results.filter(like = "sim")
    sim_test.columns = np.arange(0,len(dt_list_names))
    obs_test = results.filter(like = "obs")
    obs_test.columns = np.arange(0,len(dt_list_names))
    
    err_test = sim_test-obs_test
    err_nash = obs_test - np.mean(datafullwide[datafullwide.index < date_start_test], 
                                    axis = 0).values.reshape(-1,len(sim_test.columns))
    
    MSE_test =  np.mean((err_test) ** 2, axis = 0)
    RMSE_test = np.sqrt(np.mean((sim_test-obs_test) ** 2, axis = 0))
    
    if((sum(sim_test.isnull().any()) > 0)):
        rr_test = np.nan
    if((sum(sim_test.isnull().any()) == 0)):
        rr_test = np.zeros(len(sim_test.columns))
        for i in range(len(sim_test.columns)):
            rr_test[i] = stats.pearsonr(sim_test.iloc[:,i], obs_test.iloc[:,i])[0]
        
    NSE_test = 1 - ((np.sum(err_test ** 2, axis = 0)) / (np.sum((err_nash) ** 2, axis = 0)))
    
    Bias_test = np.mean(err_test, axis = 0)
    
    alpha_test = np.std(sim_test, axis = 0)/np.std(obs_test, axis = 0)
    beta_test = np.mean(sim_test, axis = 0)/np.mean(obs_test, axis = 0)
    KGE_test = 1-np.sqrt((rr_test-1)**2+(alpha_test-1)**2+(beta_test-1)**2)
    
    # concat scores
    scores = pd.DataFrame([NSE_test, KGE_test, rr_test**2, Bias_test, MSE_test, RMSE_test]).transpose()
    scores.index = dt_list_names
    scores.columns = ['NSE','KGE','R2','Bias','MSE','RMSE']
    
    
    #%% Export
    
    
    # export train history
    pd.DataFrame(model_history.history).to_csv(pth_out_i+'/losshistory.csv', 
                                               float_format='%.4f', sep = ";")
    
    # export scores
    scores.to_csv(pth_out_i+'/scores.csv', float_format='%.4f', sep = ";")
    
    # export obs+sim data of test period
    results.to_csv(pth_out_i+'/results.csv', float_format='%.4f', sep = ";")
    
    # plot loss curve
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.axhline(model_history.history['val_loss'][np.argmin(model_history.history['val_loss'])],
                color = "k", linewidth = 0.15)
    plt.plot(np.argmin(model_history.history['val_loss']), MSEvalmin, 
             marker='.', color='k', markersize=2)
    plt.text(HPi_epochs-1, 0.7, '\nMSE='+str("{:.4f}".format(MSEvalmin)),
             horizontalalignment = 'right', verticalalignment = 'top')
    plt.legend(['train', 'stop'], loc='upper left')
    plt.ylabel("MSE")
    plt.xlabel("timestep")
    plt.ylim(0.1,0.7)
    # plt.yscale("log")
    plt.savefig(pth_out_i+'/losscurve.png',
              dpi=300, bbox_inches='tight')
    plt.show()
    
