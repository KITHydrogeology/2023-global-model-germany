# -*- coding: utf-8 -*-
"""
@author: Benedikt Heudorfer
"""

import os
import numpy as np
import pandas as pd
# import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# import geopandas as gpd
# from scipy.spatial import cKDTree
# from scipy.stats import pearsonr

pth = "C:/Users/Benedikt Heudorfer/Documents/BGR_KNN/06a_Paper Globales Modell/"
os.chdir(pth)
pth_out = pth+"03grafiken/"

def my_ceil(a, precision=0):
    return np.ceil(a*10**(precision))*10**(-precision)

def my_floor(a, precision=0):
    return np.floor(a*10**(precision))*10**(-precision)

def my_round(a, precision=0):
    return np.round(a*10**(precision))*10**(-precision)


#%% losscurves - with and wihtout lrs - double graph


def get_losscurves(vari):
    
    pths_in = pth+"02ergebnisse/"+vari+"/"
    pths_in = [pths_in+item+'/losshistory.csv' for item in os.listdir(pths_in) if 'run' in item]
    losscurves = list()
    for i in range(len(pths_in)):
        temp = pd.read_csv(pths_in[i], sep=";", index_col=0)
        temp = pd.merge(temp, pd.DataFrame({'run':np.repeat(i,len(temp))}),
                        left_index=True,right_index=True)
        losscurves.append(temp)
    
    pths_in_nolrs = pth+"02ergebnisse/"+vari+"_nolrs/"
    pths_in_nolrs = [pths_in_nolrs+item+'/losshistory.csv' for item in os.listdir(pths_in_nolrs) if 'run' in item]
    losscurves_nolrs = list()
    for i in range(len(pths_in_nolrs)):
        temp = pd.read_csv(pths_in_nolrs[i], sep=";", index_col=0)
        temp = pd.merge(temp, pd.DataFrame({'run':np.repeat(i,len(temp))}),
                        left_index=True,right_index=True)
        losscurves_nolrs.append(temp)
    
    return(losscurves,losscurves_nolrs)

# get
losscurves_env,losscurves_nolrs_env = get_losscurves("envfeat")
losscurves_ts,losscurves_nolrs_ts = get_losscurves("tsfeat")

# some statistics
MSEvals_ts = pd.DataFrame(np.zeros((10,2)))
MSEvals_ts.columns = ["minMSE","minMSE_nolrs"]
for i in range(len(losscurves_ts)):
    MSEvals_ts.minMSE[i] = np.min(losscurves_ts[i].val_loss)
    MSEvals_ts.minMSE_nolrs[i] = np.min(losscurves_nolrs_ts[i].val_loss)
MSEvals_ts["diff"] = MSEvals_ts.minMSE-MSEvals_ts.minMSE_nolrs

MSEvals_env = pd.DataFrame(np.zeros((10,2)))
MSEvals_env.columns = ["minMSE","minMSE_nolrs"]
for i in range(len(losscurves_env)):
    MSEvals_env.minMSE[i] = np.min(losscurves_env[i].val_loss)
    MSEvals_env.minMSE_nolrs[i] = np.min(losscurves_nolrs_env[i].val_loss)
MSEvals_env["diff"] = MSEvals_env.minMSE-MSEvals_env.minMSE_nolrs

print(str(np.sum(MSEvals_ts["diff"] > 0))+"/10 in TSfeat better without lrs, with a mean of "+\
      str(my_round(np.sum(MSEvals_ts["diff"]),10)))
print(str(np.sum(MSEvals_env["diff"] > 0))+"/10 in ENVfeat better without lrs, with a mean of "+\
      str(my_round(np.sum(MSEvals_env["diff"]),10)))


fig, axs = plt.subplots(1,2,figsize=(9,4), gridspec_kw={'wspace':0})

# env
for i in range(len(losscurves_env)):
    lci_nolrs_env = losscurves_nolrs_env[i]
    lci_nolrs_env = lci_nolrs_env.set_index(np.arange(1,31))
    axs[0].plot(lci_nolrs_env.loss, color = "tab:cyan", linewidth=0.5, linestyle="dashed")
    axs[0].plot(lci_nolrs_env.val_loss, color = "#ffc30f", linewidth=0.5, linestyle="dashed")
for i in range(len(losscurves_env)):
    lci_env = losscurves_env[i]
    lci_env = lci_env.set_index(np.arange(1,31))
    axs[0].plot(lci_env.loss, color = "tab:blue", linewidth=1)
    axs[0].plot(lci_env.val_loss, color = "tab:orange", linewidth=1)
    
# ts
for i in range(len(losscurves_ts)):
    lci_nolrs_ts = losscurves_nolrs_ts[i]
    lci_nolrs_ts = lci_nolrs_ts.set_index(np.arange(1,31))
    axs[1].plot(lci_nolrs_ts.loss, color = "tab:cyan", linewidth=0.5, linestyle="dashed")
    axs[1].plot(lci_nolrs_ts.val_loss, color = "#ffc30f", linewidth=0.5, linestyle="dashed")
for i in range(len(losscurves_ts)):
    lci_ts = losscurves_ts[i]
    lci_ts = lci_ts.set_index(np.arange(1,31))
    axs[1].plot(lci_ts.loss, color = "tab:blue", linewidth=1)
    axs[1].plot(lci_ts.val_loss, color = "tab:orange", linewidth=1)

axs[0].text(30,0.48,"(a) ENVfeat", horizontalalignment="right", verticalalignment="top",
            backgroundcolor = "white")
axs[1].text(30,0.48,"(b) TSfeat", horizontalalignment="right", verticalalignment="top",
            backgroundcolor = "white")

axs[0].set_ylabel("MSE")
axs[0].set_xlabel("epoch")
axs[0].set_ylim(0.1,0.5)
axs[0].legend([Line2D([0], [0], color='tab:blue', linewidth=1),
               Line2D([0], [0], color='tab:orange', linewidth=1),
               Line2D([0], [0], color='tab:cyan', linestyle='dashed', linewidth=0.75),
               Line2D([0], [0], color='#ffc30f', linestyle='dashed', linewidth=0.75)],
              ['train', 'val', 'train (no LRS)', 'val (no LRS)'], 
              loc='upper left')
axs[1].set_xlabel("epoch")
axs[1].set_ylim(0.1,0.5)
axs[1].tick_params(left=False, labelleft=False)

fig.savefig(pth_out+'losscurves.pdf', dpi=600, bbox_inches='tight')



#%% losscurves - with and wihtout lrs - single graphs


# chose one option

vari = "tsfeat"
# vari = "envfeat"

#------------------------------------------------------------------------------

pths_in = pth+"02ergebnisse/"+vari+"/"
pths_in = [pths_in+item+'/losshistory.csv' for item in os.listdir(pths_in) if 'run' in item]
losscurves = list()
for i in range(len(pths_in)):
    temp = pd.read_csv(pths_in[i], sep=";", index_col=0)
    temp = pd.merge(temp, pd.DataFrame({'run':np.repeat(i,len(temp))}),
                    left_index=True,right_index=True)
    losscurves.append(temp)

pths_in_nolrs = pth+"02ergebnisse/"+vari+"_nolrs/"
pths_in_nolrs = [pths_in_nolrs+item+'/losshistory.csv' for item in os.listdir(pths_in_nolrs) if 'run' in item]
losscurves_nolrs = list()
for i in range(len(pths_in_nolrs)):
    temp = pd.read_csv(pths_in_nolrs[i], sep=";", index_col=0)
    temp = pd.merge(temp, pd.DataFrame({'run':np.repeat(i,len(temp))}),
                    left_index=True,right_index=True)
    losscurves_nolrs.append(temp)



fig, axs = plt.subplots(1,1,figsize=(6,5))

for i in range(len(losscurves)):
    lci_nolrs = losscurves_nolrs[i]
    axs.plot(lci_nolrs.loss, color = "tab:cyan", linewidth=0.5, linestyle="dashed")
    axs.plot(lci_nolrs.val_loss, color = "#ffc30f", linewidth=0.5, linestyle="dashed")

for i in range(len(losscurves)):
    lci = losscurves[i]
    axs.plot(lci.loss, color = "tab:blue", linewidth=1)
    axs.plot(lci.val_loss, color = "tab:orange", linewidth=1)
    
axs.set_ylabel("MSE")
axs.set_xlabel("epoch")
axs.set_ylim(0.1,0.5)
axs.legend([Line2D([0], [0], color='tab:blue', linewidth=1),
            Line2D([0], [0], color='tab:orange', linewidth=1),
            Line2D([0], [0], color='tab:cyan', linestyle='dashed', linewidth=0.75),
            Line2D([0], [0], color='#ffc30f', linestyle='dashed', linewidth=0.75)],
           ['train', 'val', 'train (no lrs)', 'val (no lrs)'], 
           loc='upper left', fontsize="small")

fig.savefig(pth_out+'losscurves_'+vari+'.pdf', dpi=600, bbox_inches='tight')
# fig.show()


#%% losscurves - appendix

def get_losscurves_2(vari):
    
    pths_in = pth+"02ergebnisse/"+vari+"/"
    pths_in = [pths_in+item+'/losshistory.csv' for item in os.listdir(pths_in) if 'run' in item]
    losscurves = list()
    for i in range(len(pths_in)):
        temp = pd.read_csv(pths_in[i], sep=";", index_col=0)
        temp = pd.merge(temp, pd.DataFrame({'run':np.repeat(i,len(temp))}),
                        left_index=True,right_index=True)
        losscurves.append(temp)
    
    return(losscurves)

# get
losscurves_rnd9 = get_losscurves_2("rndfeat9")
losscurves_rnd18 = get_losscurves_2("rndfeat18")
losscurves_dyn = get_losscurves_2("ablation_onlydyn")


fig, axs = plt.subplots(1,3,figsize=(8,3), gridspec_kw={'wspace':0})

# rnd9
for i in range(len(losscurves_rnd9)):
    lci_rnd = losscurves_rnd9[i]
    lci_rnd = lci_rnd.set_index(np.arange(1,31))
    axs[0].plot(lci_rnd.loss, color = "tab:blue", linewidth=1)
    axs[0].plot(lci_rnd.val_loss, color = "tab:orange", linewidth=1)
    
# rnd18
for i in range(len(losscurves_rnd9)):
    lci_rnd = losscurves_rnd9[i]
    lci_rnd = lci_rnd.set_index(np.arange(1,31))
    axs[1].plot(lci_rnd.loss, color = "tab:blue", linewidth=1)
    axs[1].plot(lci_rnd.val_loss, color = "tab:orange", linewidth=1)
    axs[1].set_yticks([])
    
# dyn
for i in range(len(losscurves_dyn)):
    lci_dyn = losscurves_dyn[i]
    lci_dyn = lci_dyn.set_index(np.arange(1,31))
    axs[2].plot(lci_dyn.loss, color = "tab:blue", linewidth=1)
    axs[2].plot(lci_dyn.val_loss, color = "tab:orange", linewidth=1)
    axs[2].set_yticks([])

axs[0].text(30,0.48,"(a) RNDfeat9", horizontalalignment="right", verticalalignment="top",
            backgroundcolor = "white")
axs[1].text(30,0.48,"(b) RNDfeat18", horizontalalignment="right", verticalalignment="top",
            backgroundcolor = "white")
axs[2].text(30,0.48,"(c) DNYfeat", horizontalalignment="right", verticalalignment="top",
            backgroundcolor = "white")

axs[0].set_ylabel("MSE")
axs[0].set_xlabel("epoch")
axs[1].set_xlabel("epoch")
axs[2].set_xlabel("epoch")
axs[0].set_ylim(0.1,0.5)
axs[1].set_ylim(0.1,0.5)
axs[2].set_ylim(0.1,0.5)
axs[1].set_xlim(0.01,31)
axs[2].set_xlim(0.01,31)
axs[0].legend([Line2D([0], [0], color='tab:blue', linewidth=1),
               Line2D([0], [0], color='tab:orange', linewidth=1)],
              ['train', 'val'], 
              loc='upper left')
# axs[1].tick_params(left=False, labelleft=False)

fig.savefig(pth_out+'losscurves_appendix.pdf', dpi=600, bbox_inches='tight')




#%% lrs learning rate schedule

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

# define plotting parameters
batchsize = 512
batchruns = round(160415/batchsize) # number of batch runs in 1 epoch (at 160415 samples)

# Create learning rate schedule example
steps = np.arange(0, batchruns*30, 1)
lrs = []
for step in steps:
    lrs.append(lr_warmup_cosine_decay(step, total_steps=len(steps), warmup_steps=batchruns, hold=0))

# Plot
plt.plot(lrs)
plt.xticks(np.arange(0,batchruns*31,batchruns*5), ["0","5","10","15","20","25","30"])
plt.xlabel("epochs")
plt.ylabel("learning rate")
plt.savefig(pth_out+"lrs.pdf", dpi=600, bbox_inches='tight')


#%% score plot & range


# get NSE scores
def get_scores(pth):
    pths = [pth+item+'/scores.csv' for item in os.listdir(pth) if 'run' in item]
    out = list()
    for i in range(len(pths)):
        temp = pd.read_csv(pths[i], sep=";", index_col=0)
        out.append(temp.NSE)
    out = pd.concat(out, axis=1)
    return(out)
    
def get_scores_cv(pth):
    pths = [pth+item+'/' for item in os.listdir(pth) if 'run' in item]
    allscores = list()
    for i in range(len(pths)):
        tempout = list()
        temppth = [pths[i]+item for item in os.listdir(pths[i]) if 'scores' in item]
        for j in range(len(temppth)):
            tempout.append(pd.read_csv(temppth[j], sep=';', index_col=0))
        tempout = pd.concat(tempout).sort_index()
        allscores.append(tempout)
    out = pd.concat([item.NSE for item in allscores], axis=1)
    return(out)

#------------------------------------------------------------------------------

# get scores for environmental features
pths_env = pth+"02ergebnisse/envfeat/"
NSE_env = get_scores(pths_env)
print("NSE_env="+str(np.median(NSE_env))[0:6])

# get scores for environmental features - k-fold CV
pths_env_cv = pth+"02ergebnisse/envfeat_CV/"
NSE_env_cv = get_scores_cv(pths_env_cv)
print("NSE_env_cv="+str(np.median(NSE_env_cv))[0:6])

# get scores for environmental features - CNN version
pths_env_CNN = pth+"02ergebnisse/envfeat_CNN/"
NSE_env_CNN = get_scores(pths_env_CNN)
print("NSE_env_CNN="+str(np.median(NSE_env_CNN))[0:6])

# get scores for environmental features - CNN version - k-fold CV
pths_env_CNN_cv = pth+"02ergebnisse/envfeat_CNN_CV/"
NSE_env_CNN_cv = get_scores_cv(pths_env_CNN_cv)
print("NSE_env_CNN_cv="+str(np.median(NSE_env_CNN_cv))[0:6])

# get scores for environmental features - no lrs
pths_env_nolrs = pth+"02ergebnisse/envfeat_nolrs/"
NSE_env_nolrs = get_scores(pths_env_nolrs)
print("NSE_env_nolrs="+str(np.median(NSE_env_nolrs))[0:6])

#------------------------------------------------------------------------------

# get scores for time series features
pths_ts = pth+"02ergebnisse/tsfeat/"
NSE_ts = get_scores(pths_ts)
print("NSE_ts="+str(np.median(NSE_ts))[0:6])
    
# get scores for time series features - k-fold CV
pths_ts_cv = pth+"02ergebnisse/tsfeat_CV/"
NSE_ts_cv = get_scores_cv(pths_ts_cv)
print("NSE_ts_cv="+str(np.median(NSE_ts_cv))[0:6])
    
# get scores for time series features - CNN version
pths_ts_CNN = pth+"02ergebnisse/tsfeat_CNN/"
NSE_ts_CNN = get_scores(pths_ts_CNN)
print("NSE_ts_CNN="+str(np.median(NSE_ts_CNN))[0:6])

# get scores for time series features - CNN version - k-fold CV
pths_ts_CNN_cv = pth+"02ergebnisse/tsfeat_CNN_CV/"
NSE_ts_CNN_cv = get_scores_cv(pths_ts_CNN_cv)
print("NSE_ts_CNN_cv="+str(np.median(NSE_ts_CNN_cv))[0:6])
    
# get scores for environmental features - no lrs
pths_ts_nolrs = pth+"02ergebnisse/tsfeat_nolrs/"
NSE_ts_nolrs = get_scores(pths_ts_nolrs)
print("NSE_ts_nolrs="+str(np.median(NSE_ts_nolrs))[0:6])

#------------------------------------------------------------------------------

# get scores for only dynamic features
pths_ablation = pth+"02ergebnisse/ablation_onlydyn/"
NSE_ablation = get_scores(pths_ablation)
print("NSE_ablation="+str(np.median(NSE_ablation))[0:6])

# get scores for only dynamic features - k-fold CV
pths_ablation_cv = pth+"02ergebnisse/ablation_onlydyn_CV/"
NSE_ablation_cv = get_scores_cv(pths_ablation_cv)
print("NSE_ablation_cv="+str(np.median(NSE_ablation_cv))[0:6])

# get scores for random features - 9
pths_rnd9 = pth+"02ergebnisse/rndfeat9/"
NSE_rnd9 = get_scores(pths_rnd9)
print("NSE_rnd9="+str(np.median(NSE_rnd9))[0:6])

# get scores for random features - 9 - k-fold CV
pths_rnd9_cv = pth+"02ergebnisse/rndfeat9_CV/"
NSE_rnd9_cv = get_scores_cv(pths_rnd9_cv)
print("NSE_rnd9_cv="+str(np.median(NSE_rnd9_cv))[0:6])

# get scores for random features - 18
pths_rnd18 = pth+"02ergebnisse/rndfeat18/"
NSE_rnd18 = get_scores(pths_rnd18)
print("NSE_rnd18="+str(np.median(NSE_rnd18))[0:6])

# get scores for random features - 18
pths_rnd18_cv = pth+"02ergebnisse/rndfeat18_cv/"
NSE_rnd18_cv = get_scores_cv(pths_rnd18_cv)
print("NSE_rnd18_cv="+str(np.median(NSE_rnd18_cv))[0:6])


# get benchmark scores
filst = os.listdir(pth+'01data/benchmark')
filst_IDob = [item in NSE_env.index for item in [item[7:-4] for item in filst]]
filst_IDob = [i for i,sublist in enumerate(filst_IDob) if sublist]
filst = [filst[item] for item in filst_IDob]; del filst_IDob
benchmark = list()
for item in filst:
    benchmark.append(pd.read_csv(pth+'01data/benchmark/'+item, sep=";", index_col=0))
benchmark = pd.concat(benchmark)
benchmark.index = [item[7:-4] for item in filst]
print("NSE_bench="+str(np.median(benchmark.NSE))[0:6])


#------------------------------------------------------------------------------


# table with all scores
fullscores = pd.DataFrame({"single_well":benchmark.NSE,
                           "ENVfeat_IS":NSE_env.median(axis=1),
                           "TSfeat_IS":NSE_ts.median(axis=1),
                           "ENVfeat_OOS":NSE_env_cv.median(axis=1),
                           "TSfeat_OOS":NSE_ts_cv.median(axis=1),
                           "RNDfeat9_IS":NSE_rnd9.median(axis=1),
                           "RNDfeat9_OOS":NSE_rnd9_cv.median(axis=1),
                           "RNDfeat18_IS":NSE_rnd18.median(axis=1),
                           "RNDfeat18_OOS":NSE_rnd18_cv.median(axis=1),
                           "DYNonly_IS":NSE_ablation.median(axis=1),
                           "DYNonly_OOS":NSE_ablation_cv.median(axis=1)
                           })
fullscores.to_csv(pth_out+"table_fullscores.csv", sep=";")

# table with mean scores
meanscores = pd.DataFrame({"model":["single","env","ts","rnd9","rnd18","ablation",
                                    "env_cv","ts_cv","rnd9_cv","rnd18_cv","ablation_cv"],
                            "NSE_lower":[0,
                                         np.mean(NSE_env.quantile(0.1, axis=1)),
                                         np.mean(NSE_ts.quantile(0.1, axis=1)),
                                         np.mean(NSE_rnd9.quantile(0.1, axis=1)),
                                         np.mean(NSE_rnd18.quantile(0.1, axis=1)),
                                         np.mean(NSE_ablation.quantile(0.1, axis=1)),
                                         np.mean(NSE_env_cv.quantile(0.1, axis=1)),
                                         np.mean(NSE_ts_cv.quantile(0.1, axis=1)),
                                         np.mean(NSE_rnd9_cv.quantile(0.1, axis=1)),
                                         np.mean(NSE_rnd18_cv.quantile(0.1, axis=1)),
                                         np.mean(NSE_ablation_cv.quantile(0.1, axis=1))],
                            "NSE_mean":[np.mean(benchmark.NSE),
                                        np.mean(NSE_env.median(axis=1)),
                                        np.mean(NSE_ts.median(axis=1)),
                                        np.mean(NSE_rnd9.median(axis=1)),
                                        np.mean(NSE_rnd18.median(axis=1)),
                                        np.mean(NSE_ablation.median(axis=1)),
                                        np.mean(NSE_env_cv.median(axis=1)),
                                        np.mean(NSE_ts_cv.median(axis=1)),
                                        np.mean(NSE_rnd9_cv.median(axis=1)),
                                        np.mean(NSE_rnd18_cv.median(axis=1)),
                                        np.mean(NSE_ablation_cv.median(axis=1))],
                            "NSE_upper":[0,
                                         np.mean(NSE_env.quantile(0.9, axis=1)),
                                         np.mean(NSE_ts.quantile(0.9, axis=1)),
                                         np.mean(NSE_rnd9.quantile(0.9, axis=1)),
                                         np.mean(NSE_rnd18.quantile(0.9, axis=1)),
                                         np.mean(NSE_ablation.quantile(0.9, axis=1)),
                                         np.mean(NSE_env_cv.quantile(0.9, axis=1)),
                                         np.mean(NSE_ts_cv.quantile(0.9, axis=1)),
                                         np.mean(NSE_rnd9_cv.quantile(0.9, axis=1)),
                                         np.mean(NSE_rnd18_cv.quantile(0.9, axis=1)),
                                         np.mean(NSE_ablation_cv.quantile(0.9, axis=1))]
                            })
meanscores.to_csv(pth_out+"table_meanscores.csv", sep=";", index=False)

#%%

#------------------------------------------------------------------------------
# plot scoreplot

fig, axs = plt.subplots(1,1,figsize=(9,5))#, gridspec_kw={'wspace':0})

alphabub = 0.11
plotpos = (np.arange(len(NSE_env))/(len(NSE_env)-1))
plotpos_fill = pd.concat([pd.Series(plotpos), pd.Series(np.flip(plotpos))])

# NSE environmental features
axs.fill(pd.concat([NSE_env.quantile(0.9, axis=1).sort_values(),
                    NSE_env.quantile(0.1, axis=1).sort_values(ascending = False)]),
         plotpos_fill, color='crimson', alpha=alphabub)
axs.plot(NSE_env.median(axis=1).sort_values(), plotpos, color="crimson")

# NSE environmental features - k-fold CV
axs.fill(pd.concat([NSE_env_cv.quantile(0.9, axis=1).sort_values(),
                    NSE_env_cv.quantile(0.1, axis=1).sort_values(ascending = False)]),
         plotpos_fill, color='crimson', alpha=alphabub)
axs.plot(NSE_env_cv.median(axis=1).sort_values(), plotpos, linestyle="dashed", color="crimson")

# NSE time series features
axs.fill(pd.concat([NSE_ts.quantile(0.9, axis=1).sort_values(),
                    NSE_ts.quantile(0.1, axis=1).sort_values(ascending = False)]),
         plotpos_fill, color='blue', alpha=alphabub)
axs.plot(NSE_ts.median(axis=1).sort_values(), plotpos, color="blue")

# NSE time series features - k-fold CV
axs.fill(pd.concat([NSE_ts_cv.quantile(0.9, axis=1).sort_values(),
                    NSE_ts_cv.quantile(0.1, axis=1).sort_values(ascending = False)]),
         plotpos_fill, color='blue', alpha=alphabub)
axs.plot(NSE_ts_cv.median(axis=1).sort_values(), plotpos, linestyle="dashed", color="blue")

# NSE random - 9
axs.fill(pd.concat([NSE_rnd9.quantile(0.9, axis=1).sort_values(),
                    NSE_rnd9.quantile(0.1, axis=1).sort_values(ascending = False)]),
         plotpos_fill, color='dodgerblue', alpha=alphabub)
axs.plot(NSE_rnd9.median(axis=1).sort_values(), plotpos, color="dodgerblue")

# NSE random - 9 - k-fold CV
axs.fill(pd.concat([NSE_rnd9_cv.quantile(0.9, axis=1).sort_values(),
                    NSE_rnd9_cv.quantile(0.1, axis=1).sort_values(ascending = False)]),
         plotpos_fill, color='dodgerblue', alpha=alphabub)
axs.plot(NSE_rnd9_cv.median(axis=1).sort_values(), plotpos, linestyle="dashed", color="dodgerblue")

# NSE random - 18
axs.fill(pd.concat([NSE_rnd18.quantile(0.9, axis=1).sort_values(),
                    NSE_rnd18.quantile(0.1, axis=1).sort_values(ascending = False)]),
         plotpos_fill, color='tab:orange', alpha=alphabub)
axs.plot(NSE_rnd18.median(axis=1).sort_values(), plotpos, color="tab:orange")

# NSE random - 18 - k-fold CV
axs.fill(pd.concat([NSE_rnd18_cv.quantile(0.9, axis=1).sort_values(),
                    NSE_rnd18_cv.quantile(0.1, axis=1).sort_values(ascending = False)]),
         plotpos_fill, color='tab:orange', alpha=alphabub)
axs.plot(NSE_rnd18_cv.median(axis=1).sort_values(), plotpos, 
         linestyle="dashed", color="tab:orange")

# NSE ablation
axs.fill(pd.concat([NSE_ablation.quantile(0.9, axis=1).sort_values(),
                    NSE_ablation.quantile(0.1, axis=1).sort_values(ascending = False)]),
         plotpos_fill, color='blueviolet', alpha=alphabub-0.02)
axs.plot(NSE_ablation.median(axis=1).sort_values(), plotpos, 
         color="blueviolet", alpha=0.8)

# NSE ablation - k-fold CV
axs.fill(pd.concat([NSE_ablation_cv.quantile(0.9, axis=1).sort_values(),
                    NSE_ablation_cv.quantile(0.1, axis=1).sort_values(ascending = False)]),
         plotpos_fill, color='blueviolet', alpha=alphabub-0.02)
axs.plot(NSE_ablation_cv.median(axis=1).sort_values(), plotpos, 
         linestyle="dashed", color="blueviolet", alpha=0.8)

# NSE benchmark
axs.plot(benchmark.NSE.sort_values(), plotpos, color="k", linewidth=1)


axs.set_xlim(0.4,0.96)
axs.set_xlabel("NSE")
axs.set_ylabel("CDF")

axs.legend([Line2D([0], [0], color='black'),
            Line2D([0], [0], color='crimson'),
            Line2D([0], [0], color='crimson', linestyle='dashed'),
            Line2D([0], [0], color='tab:orange'),
            Line2D([0], [0], color='tab:orange', linestyle='dashed'),
            Line2D([0], [0], color='blue'),
            Line2D([0], [0], color='blue', linestyle='dashed'),
            Line2D([0], [0], color='dodgerblue'),
            Line2D([0], [0], color='dodgerblue', linestyle='dashed'),
            Line2D([0], [0], color='blueviolet', alpha=0.8),
            Line2D([0], [0], color='blueviolet', linestyle='dashed', alpha=0.8)],
           ['single-well*',
            'ENVfeat (IS)','ENVfeat (OOS)',
            'RNDfeat18 (IS)','RNDfeat18 (OOS)',
            'TSfeat (IS)','TSfeat (OOS)',
            'RNDfeat9 (IS)','RNDfeat9 (OOS)',
            'DYNonly (IS)','DYNonly (OOS)'])

fig.savefig(pth_out+'/scoreplot.pdf', dpi=600, bbox_inches='tight')
# fig.show()

#------------------------------------------------------------------------------
# plot scoreplot without ranges

fig, axs = plt.subplots(1,1,figsize=(9,5))#, gridspec_kw={'wspace':0})

alphabub = 0.11

# NSE environmental features
axs.plot(NSE_env.median(axis=1).sort_values(), plotpos, color="crimson")

# NSE environmental features - k-fold CV
axs.plot(NSE_env_cv.median(axis=1).sort_values(), plotpos, linestyle="dashed", color="crimson")

# NSE time series features
axs.plot(NSE_ts.median(axis=1).sort_values(), plotpos, color="blue")

# NSE time series features - k-fold CV
axs.plot(NSE_ts_cv.median(axis=1).sort_values(), plotpos, linestyle="dashed", color="blue")

# NSE random - 9
axs.plot(NSE_rnd9.median(axis=1).sort_values(), plotpos, color="dodgerblue")

# NSE random - 9 - k-fold CV
axs.plot(NSE_rnd9_cv.median(axis=1).sort_values(), plotpos, linestyle="dashed", color="dodgerblue")

# NSE random - 18
axs.plot(NSE_rnd18.median(axis=1).sort_values(), plotpos, color="tab:orange")

# NSE random - 18 - k-fold CV
axs.plot(NSE_rnd18_cv.median(axis=1).sort_values(), plotpos, linestyle="dashed", color="tab:orange")

# NSE ablation
axs.plot(NSE_ablation.median(axis=1).sort_values(), plotpos, color="blueviolet", alpha=0.8)

# NSE ablation - k-fold CV
axs.plot(NSE_ablation_cv.median(axis=1).sort_values(), plotpos, linestyle="dashed", color="blueviolet", alpha=0.8)

# NSE benchmark
axs.plot(benchmark.NSE.sort_values(), plotpos, color="k", linewidth=1)

axs.set_xlim(0.4,0.96)
axs.set_xlabel("NSE")
axs.set_ylabel("CDF")

axs.legend([Line2D([0], [0], color='black'),
            Line2D([0], [0], color='crimson'),
            Line2D([0], [0], color='crimson', linestyle='dashed'),
            Line2D([0], [0], color='tab:orange'),
            Line2D([0], [0], color='tab:orange', linestyle='dashed'),
            Line2D([0], [0], color='blue'),
            Line2D([0], [0], color='blue', linestyle='dashed'),
            Line2D([0], [0], color='dodgerblue'),
            Line2D([0], [0], color='dodgerblue', linestyle='dashed'),
            Line2D([0], [0], color='blueviolet', alpha=0.8),
            Line2D([0], [0], color='blueviolet', linestyle='dashed', alpha=0.8)],
           ['single-well*',
            'ENVfeat (IS)','ENVfeat (OOS)',
            'RNDfeat18 (IS)','RNDfeat18 (OOS)',
            'TSfeat (IS)','TSfeat (OOS)',
            'RNDfeat9 (IS)','RNDfeat9 (OOS)',
            'DYNonly (IS)','DYNonly (OOS)'])

fig.savefig(pth_out+'/scoreplot_without_ranges.pdf', dpi=600, bbox_inches='tight')
# fig.show()


#------------------------------------------------------------------------------
# scoreplot comparison CNN 

fig, axs = plt.subplots(1,1,figsize=(9,5), gridspec_kw={'wspace':0})

alphabub = 0.11

# NSE environmental features - LSTM
axs.fill(pd.concat([NSE_env.quantile(0.9, axis=1).sort_values(),
                    NSE_env.quantile(0.1, axis=1).sort_values(ascending = False)]),
         plotpos_fill, color='crimson', alpha=alphabub)
axs.plot(NSE_env.median(axis=1).sort_values(), plotpos, color="crimson")

# NSE environmental features - LSTM - k-fold CV
axs.fill(pd.concat([NSE_env_cv.quantile(0.9, axis=1).sort_values(),
                    NSE_env_cv.quantile(0.1, axis=1).sort_values(ascending = False)]),
         plotpos_fill, color='crimson', alpha=alphabub)
axs.plot(NSE_env_cv.median(axis=1).sort_values(), plotpos, linestyle="dashed", color="crimson")

# NSE environmental features - CNN
axs.fill(pd.concat([NSE_env_CNN.quantile(0.9, axis=1).sort_values(),
                    NSE_env_CNN.quantile(0.1, axis=1).sort_values(ascending = False)]),
         plotpos_fill, color='tab:cyan', alpha=alphabub)
axs.plot(NSE_env_CNN.median(axis=1).sort_values(), plotpos, color="tab:cyan")

# NSE environmental features - CNN - k-fold CV
axs.fill(pd.concat([NSE_env_CNN_cv.quantile(0.9, axis=1).sort_values(),
                    NSE_env_CNN_cv.quantile(0.1, axis=1).sort_values(ascending = False)]),
         plotpos_fill, color='tab:cyan', alpha=alphabub)
axs.plot(NSE_env_CNN_cv.median(axis=1).sort_values(), plotpos, color="tab:cyan", linestyle="dashed")

axs.set_xlim(0.4,0.96)
axs.set_xlabel("NSE")
axs.set_ylabel("CDF")

axs.legend([Line2D([0], [0], color='crimson'),
            Line2D([0], [0], color='crimson', linestyle='dashed'),
            Line2D([0], [0], color='tab:cyan'),
            Line2D([0], [0], color='tab:cyan', linestyle='dashed')],
            ['ENVfeat - LSTM (IS)','ENVfeat - LSTM (OOS)',
             'ENVfeat - CNN (IS)','ENVfeat - CNN (OOS)'])

fig.savefig(pth_out+'/scoreplot_compareCNN.pdf', dpi=600, bbox_inches='tight')
# fig.show()


#------------------------------------------------------------------------------
# score ranges

plotty = [NSE_env.max(axis=1)-NSE_env.min(axis=1),
           NSE_rnd18.max(axis=1)-NSE_rnd18.min(axis=1),
           NSE_ts.max(axis=1)-NSE_ts.min(axis=1),
           NSE_rnd9.max(axis=1)-NSE_rnd9.min(axis=1),
           NSE_ablation.max(axis=1)-NSE_ablation.min(axis=1),
            NSE_env_cv.max(axis=1)-NSE_env_cv.min(axis=1),
            NSE_rnd18_cv.max(axis=1)-NSE_rnd18_cv.min(axis=1),
            NSE_ts_cv.max(axis=1)-NSE_ts_cv.min(axis=1),
            NSE_rnd9_cv.max(axis=1)-NSE_rnd9_cv.min(axis=1),
            NSE_ablation_cv.max(axis=1)-NSE_ablation_cv.min(axis=1)
           ]

plt.figure(figsize=(10,4))
violin = plt.violinplot(plotty, widths=.7, showextrema=False)
for pc in violin['bodies']:
    pc.set_facecolor('dodgerblue')
plt.boxplot(plotty, widths=.7, medianprops = dict(color = "k"))
plt.plot([5.5,5.5],[np.min(plotty),np.max(plotty)], color="k", linewidth=0.5)
plt.xticks([1,2,3,4,5,6,7,8,9,10],
           ['ENVfeat','RNDfeat18','TSfeat','RNDfeat9','DYNonly',
            'ENVfeat','RNDfeat18','TSfeat','RNDfeat9','DYNonly'])
plt.text(1, np.max(plotty), "(IS)", verticalalignment="top", size=12)
plt.text(10, np.max(plotty), "(OOS)", verticalalignment="top", horizontalalignment="right", size=12)
plt.yscale("log")
plt.yticks([0.01,0.05,0.1,0.5,1],[0.01,0.05,0.1,0.5,1])
plt.tick_params(right=True, labelright=True)
plt.tick_params(right=True, labelright=True, which="minor")
# plt.ylim(0,1.5)
plt.ylabel('Range of NSE scores (10 seeds)')
plt.savefig(pth_out+'/scorerange.pdf', dpi=600, bbox_inches='tight')


#%% Plot PFI

fig = plt.figure(figsize=(9,9), dpi=600)

grid = plt.GridSpec(42, 9)

a1 = fig.add_subplot(grid[0:19, 0:4])
a2 = fig.add_subplot(grid[0:19, 5:9])
a3 = fig.add_subplot(grid[23:33, 0:4])
a4 = fig.add_subplot(grid[23:33, 5:9])
a5 = fig.add_subplot(grid[37:42, 0:9])


#------------------------------------------------------------------------------

vari1 = "envfeat"
n_dyn = 4
n_stat = 18
xlimstat = [0.14,0.22]

# static features
df_stat = pd.read_csv(pth+'/02ergebnisse/'+vari1+'/PFI_stat.csv', sep=";")
df_stat = df_stat.replace(["clc2018","dwd_percfrost","twi","buek250_bt","hyraum_gr",
                           "eumohp_lp3","eumohp_dsd3","dwd_psum","swr1000_swr","dwd_tmean",
                           "dwd_rhmean","dwd_epot","huek250_kf","hengl_sand","gwn_recharge",
                           "eumohp_sd3","hengl_clay","huek250_ha"], 
                          ["CLC land cover","frost days","TWI","soil depth","hygeo division",
                           "lateral position","divide to stream","Psum","percolation","Tmean",
                           "rHmean","ETPpot","conductivity","sand (%)","recharge",
                           "stream distance","clay (%)","aquifer type"])
df_stat = df_stat.pivot(index="seed",columns="feature", values="mse")
df_stat = df_stat[df_stat.median().sort_values(ascending=True).index]
mse_baseline = df_stat.baseline.median()

# plot
a1.boxplot(df_stat, vert=False, patch_artist=True, widths=0.7,
            boxprops=dict(facecolor="#33aaff"),
            medianprops=dict(color="k"),
            flierprops =dict(color="dodgerblue"))
a1.plot([mse_baseline,mse_baseline],[1,n_stat+1], ':', color='k',
          label=f'baseline\nMSE={mse_baseline:.3f}')
a1.set_yticks(np.arange(1,n_stat+2),df_stat.columns)
a1.set_title('static environmental features')
a1.set_xlim(xlimstat[0],xlimstat[1])


#------------------------------------------------------------------------------

vari2 = "rndfeat18"
n_dyn = 4
n_stat = 18
xlimstat = [0.14,0.22]

# static features
df_stat = pd.read_csv(pth+'/02ergebnisse/'+vari2+'/PFI_stat.csv', sep=";")
df_stat = df_stat.pivot(index="seed",columns="feature", values="mse")
df_stat = df_stat[df_stat.median().sort_values(ascending=True).index]
mse_baseline = df_stat.baseline.median()

# plot
a2.boxplot(df_stat, vert=False, patch_artist=True, widths=0.7,
            boxprops=dict(facecolor="#33aaff"),
            medianprops=dict(color="k"),
            flierprops =dict(color="dodgerblue"))
a2.plot([mse_baseline,mse_baseline],[1,n_stat+1], ':', color='k',
          label=f'baseline\nMSE={mse_baseline:.3f}')
a2.set_yticks(np.arange(1,n_stat+2),df_stat.columns)
a2.set_title('static random (18)')
a2.set_xlim(xlimstat[0],xlimstat[1])


#------------------------------------------------------------------------------

vari3 = "tsfeat"
n_dyn = 4
n_stat = 9
xlimstat = [0.14,0.22]

# static features
df_stat = pd.read_csv(pth+'/02ergebnisse/'+vari3+'/PFI_stat.csv', sep=";")
df_stat = df_stat.replace(["hpd","per","sk","seas_behav","long_rec","rr","SD_dif"], 
                          ["HPD","P52","skew","SB","LRec","RR","SDdiff"])
df_stat = df_stat.pivot(index="seed",columns="feature", values="mse")
df_stat = df_stat[df_stat.median().sort_values(ascending=True).index]
mse_baseline = df_stat.baseline.median()

# plot
a3.boxplot(df_stat, vert=False, patch_artist=True, widths=0.7,
            boxprops=dict(facecolor="#33aaff"),
            medianprops=dict(color="k"),
            flierprops =dict(color="dodgerblue"))
a3.plot([mse_baseline,mse_baseline],[1,n_stat+1], ':', color='k',
          label=f'baseline\nMSE={mse_baseline:.3f}')
a3.set_yticks(np.arange(1,n_stat+2),df_stat.columns)
a3.set_title('static time series')
a3.set_xlim(xlimstat[0],xlimstat[1])


#------------------------------------------------------------------------------

vari4 = "rndfeat9"
n_dyn = 4
n_stat = 9
xlimstat = [0.14,0.22]

# static features
df_stat = pd.read_csv(pth+'/02ergebnisse/'+vari4+'/PFI_stat.csv', sep=";")
df_stat = df_stat.pivot(index="seed",columns="feature", values="mse")
df_stat = df_stat[df_stat.median().sort_values(ascending=True).index]
mse_baseline = df_stat.baseline.median()

# plot
a4.boxplot(df_stat, vert=False, patch_artist=True, widths=0.7,
            boxprops=dict(facecolor="#33aaff"),
            medianprops=dict(color="k"),
            flierprops =dict(color="dodgerblue"))
a4.plot([mse_baseline,mse_baseline],[1,n_stat+1], ':', color='k',
          label=f'baseline\nMSE={mse_baseline:.3f}')
a4.set_yticks(np.arange(1,n_stat+2),df_stat.columns)
a4.set_title('static random (9)')
a4.set_xlim(xlimstat[0],xlimstat[1])


#------------------------------------------------------------------------------

# PFI  dynamic features for both models

n_dyn = 4

# load & merge
df_dyn1 = pd.read_csv(pth+'/02ergebnisse/'+vari1+'/PFI_dyn.csv', sep=";")
df_dyn2 = pd.read_csv(pth+'/02ergebnisse/'+vari2+'/PFI_dyn.csv', sep=";")
df_dyn3 = pd.read_csv(pth+'/02ergebnisse/'+vari3+'/PFI_dyn.csv', sep=";")
df_dyn4 = pd.read_csv(pth+'/02ergebnisse/'+vari4+'/PFI_dyn.csv', sep=";")
df_dyn = pd.concat([df_dyn1,df_dyn2,df_dyn3,df_dyn4])

# differentiate
df_dyn.seed[50:100] = df_dyn.seed[50:100]+1 
df_dyn.seed[100:150] = df_dyn.seed[100:150]+2 
df_dyn.seed[150:200] = df_dyn.seed[150:200]+3 
df_dyn = df_dyn.reset_index(drop=True)


df_dyn = df_dyn.pivot(index="seed",columns="feature", values="mse")
df_dyn = df_dyn[df_dyn.median().sort_values(ascending=True).index]
mse_baseline = df_dyn.baseline.median()

a5.boxplot(df_dyn, vert=False, patch_artist=True, widths=0.7,
            boxprops=dict(facecolor="#33aaff"),
            medianprops=dict(color="k"),
            flierprops =dict(color="dodgerblue"))
a5.plot([mse_baseline,mse_baseline],[1,n_dyn+1], ':', color='k',
          label=f'baseline\nMSE={mse_baseline:.3f}')
a5.set_xscale('log')
a5.set_yticks(np.arange(1,n_dyn+2),df_dyn.columns)
a5.set_title('dynamic features')
a5.set_xlabel("MSE with feature permuted")

fig.savefig(pth_out+'/PFI.pdf', dpi=600, bbox_inches='tight')
fig.show()

# #------------------------------------------------------------------------------


#%% time series and score table for supplements


def get_obs(vari, ii):
    pths_in = pth+"02ergebnisse/"+vari+"/"
    pths_in = [pths_in+item+'/results.csv' for item in os.listdir(pths_in) if 'run' in item]
    output = pd.read_csv(pths_in[0], sep=";", index_col=0)
    output = output.filter(like = "obs")
    output.columns = [item[:-4] for item in output.columns]
    output = pd.DataFrame(output.iloc[:,ii])
    output.index = pd.date_range("20120102", "20151228", freq='W-MON')
    return(output)

def get_sim(vari, ii):
    pths_in = pth+"02ergebnisse/"+vari+"/"
    pths_in = [pths_in+item+'/results.csv' for item in os.listdir(pths_in) if 'run' in item]
    output = list()
    for i in range(len(pths_in)):
        temp = pd.read_csv(pths_in[i], sep=";", index_col=0)
        temp = temp.filter(like="sim")
        temp.columns = [item[:-4] for item in temp.columns]
        output.append(temp.iloc[:,ii])
    output = pd.concat(output,axis=1)
    output = pd.DataFrame(output.median(axis=1))
    output.columns = [temp.columns[ii]]  
    output.index = pd.date_range("20120102", "20151228", freq='W-MON')
    return(output)


def get_obs_sim_cv(vari, ii):
    
    pths_in = pth+"02ergebnisse/"+vari+"/"
    pths = [pths_in+item for item in os.listdir(pths_in)]
    allobs = list()
    allsim = list()
    
    for i in range(len(pths)):
        
        allresults_i = list()
        pths_i = [pths[i]+"/"+item2 for item2 in [item for item in os.listdir(pths[i]) if 'results' in item]]
        for x in pths_i: 
            allresults_i.append(pd.read_csv(x, sep=";", index_col=0))
        allresults_i = pd.concat(allresults_i,axis=1)
        
        obs_i = allresults_i.filter(like="obs")
        obs_i.columns = [item[:-4] for item in obs_i.columns]
        obs_i = obs_i.sort_index(axis = 1)
        allobs.append(pd.DataFrame(obs_i.iloc[:,ii]))
        allobs[i].index = pd.date_range("20120102", "20151228", freq='W-MON')
        
        sim_i = allresults_i.filter(like="sim")
        sim_i.columns = [item[:-4] for item in sim_i.columns]
        sim_i = sim_i.sort_index(axis = 1)
        allsim.append(pd.DataFrame(sim_i.iloc[:,ii]))
        allsim[i].index = pd.date_range("20120102", "20151228", freq='W-MON')
        
    obs_out = pd.concat(allobs, axis=1)
    obs_out = pd.DataFrame(obs_out.median(axis=1))
    obs_out.columns = [allobs[0].columns[0]]  
    sim_out = pd.concat(allsim, axis=1)
    sim_out = pd.DataFrame(sim_out.median(axis=1))
    sim_out.columns = [allsim[0].columns[0]]  
    return(obs_out, sim_out)


fullscores = pd.read_csv(pth_out+"table_fullscores.csv",sep=";",index_col=0)

# from matplotlib.backends.backend_pdf import PdfPages

# with PdfPages('test_period.pdf') as pdf:

for i in range(len(fullscores)):
    # get data for time series i 
    obs = get_obs("envfeat",i)
    sim_env = get_sim("envfeat",i)
    sim_ts = get_sim("tsfeat",i)
    sim_rnd9 = get_sim("rndfeat9",i)
    sim_rnd18 = get_sim("rndfeat18",i)
    sim_dyn = get_sim("ablation_onlydyn",i)
    obs, sim_env_cv = get_obs_sim_cv("envfeat_cv",i)
    obs, sim_ts_cv = get_obs_sim_cv("tsfeat_cv",i)
    obs, sim_rnd9_cv = get_obs_sim_cv("rndfeat9_cv",i)
    obs, sim_rnd18_cv = get_obs_sim_cv("rndfeat18_cv",i)
    obs, sim_dyn_cv = get_obs_sim_cv("ablation_onlydyn_cv",i)

    fig, axs = plt.subplots(2,1,figsize=(12,6), gridspec_kw={'hspace':0})
    
    axs[0].plot(sim_env, "crimson")
    axs[0].plot(sim_ts, "blue")
    axs[0].plot(sim_rnd9, "dodgerblue")
    axs[0].plot(sim_rnd18, "tab:orange")
    axs[0].plot(sim_dyn, "blueviolet")
    axs[0].plot(obs, "k")
    axs[0].text(0.97,0.95,"(IS)", transform=axs[0].transAxes,
                horizontalalignment="right",verticalalignment="top")
    axs[1].set_xlabel("Date")
    axs[0].set_ylabel("GWL (masl)")
    axs[1].set_ylabel("GWL (masl)")
    axs[0].set_title(str(fullscores.index[i]))

    axs[1].plot(sim_env_cv, "crimson")
    axs[1].plot(sim_ts_cv, "blue")
    axs[1].plot(sim_rnd9_cv, "dodgerblue")
    axs[1].plot(sim_rnd18_cv, "tab:orange")
    axs[1].plot(sim_dyn_cv, "blueviolet")
    axs[1].plot(obs, "k")
    axs[1].text(0.98,0.95,"(OOS)", transform=axs[1].transAxes,
                horizontalalignment="right",verticalalignment="top")
    
    axs[0].legend([Line2D([0], [0], color='k', linewidth=1.5),
                   Line2D([0], [0], color='crimson', linewidth=1.5),
                   Line2D([0], [0], color='tab:orange', linewidth=1.5),
                   Line2D([0], [0], color='blue', linewidth=1.5),
                   Line2D([0], [0], color='dodgerblue', linewidth=1.5),
                   Line2D([0], [0], color='blueviolet', linewidth=1.5)],
                  ['obs',
                   'ENVfeat (IS) NSE=%.3f' %fullscores.ENVfeat_IS[i],
                   'RNDfeat18 (IS) NSE=%.3f' %fullscores.RNDfeat18_IS[i],
                   'TSfeat (IS)   NSE=%.3f' %fullscores.TSfeat_IS[i],
                   'RNDfeat9 (IS) NSE=%.3f' %fullscores.RNDfeat9_IS[i],
                   'DYNonly (IS) NSE=%.3f' %fullscores.DYNonly_IS[i]], 
                  loc='upper center', bbox_to_anchor=(1.138,1)).get_frame().set_alpha(None)
    
    axs[1].legend([Line2D([0], [0], color='k', linewidth=1.5),
                   Line2D([0], [0], color='crimson', linewidth=1.5),
                   Line2D([0], [0], color='tab:orange', linewidth=1.5),
                   Line2D([0], [0], color='blue', linewidth=1.5),
                   Line2D([0], [0], color='dodgerblue', linewidth=1.5),
                   Line2D([0], [0], color='blueviolet', linewidth=1.5)],
                  ['obs',
                   'ENVfeat (OOS) NSE=%.3f' %fullscores.ENVfeat_OOS[i],
                   'RNDfeat18 (OOS) NSE=%.3f' %fullscores.RNDfeat18_OOS[i],
                   'TSfeat (OOS)   NSE=%.3f' %fullscores.TSfeat_OOS[i],
                   'RNDfeat8 (OOS) NSE=%.3f' %fullscores.RNDfeat9_OOS[i],
                   'DYNonly (OOS) NSE=%.3f' %fullscores.DYNonly_OOS[i]], 
                  loc='upper center', bbox_to_anchor=(1.15,1)).get_frame().set_alpha(None)
    
    # pdf.savefig(bbox_extra_artists=(legend0,legend1,))
    fig.savefig(pth_out+'/time_series/'+fullscores.index[i]+'.pdf', dpi=600, bbox_inches='tight')
    
