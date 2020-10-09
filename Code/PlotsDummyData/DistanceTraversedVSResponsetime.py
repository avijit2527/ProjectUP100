#!/usr/bin/env python
# coding: utf-8
# @author Avijit Roy


# In[ ]:

# Importing the required module
import pandas as pd
import matplotlib.image as image
import matplotlib.cbook as cbook
from matplotlib import style
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import random
import pickle
import time
import os
import math
import numpy as np
import datetime
from datetime import timedelta
import glob



import matplotlib
# matplotlib.use("Agg")

def plotDiagrams():
    no_of_iter = 25;
    x = datetime.datetime.now();
    now = str(x)[0:10];

    plt.rc('xtick',labelsize=10)
    plt.rc('ytick',labelsize=10)
    fig, ax = plt.subplots()

    coverage_array_for_alpha_over_multiple_runs = np.load("./DistanceTraversedVSResponseTime_0.npy")
    mean_coverage_array = np.mean(coverage_array_for_alpha_over_multiple_runs, axis = 0)
    std_coverage_array = np.std(coverage_array_for_alpha_over_multiple_runs,axis = 0)
    error = ((std_coverage_array.T[1] * 1.96) / math.sqrt(no_of_iter));
  

    #Plotting coverage vs number of agents
    ax.plot(mean_coverage_array.T[0],mean_coverage_array.T[1], label='Step Penalty = 0',  linestyle='solid', color='red', linewidth=6)
    ax.errorbar(mean_coverage_array.T[0],mean_coverage_array.T[1],fmt="none", Color='red', yerr=error, ecolor='red');
  

    coverage_array_for_alpha_over_multiple_runs = np.load("./DistanceTraversedVSResponseTime_-1.npy")
    mean_coverage_array = np.mean(coverage_array_for_alpha_over_multiple_runs, axis = 0)
    std_coverage_array = np.std(coverage_array_for_alpha_over_multiple_runs,axis = 0)
    error = ((std_coverage_array.T[1] * 1.96) / math.sqrt(no_of_iter));
  

    #Plotting coverage vs number of agents
    ax.plot(mean_coverage_array.T[0],mean_coverage_array.T[1], label='Step Penalty = -1',  linestyle='dotted', color='green', linewidth=6)
    ax.errorbar(mean_coverage_array.T[0],mean_coverage_array.T[1], fmt="none", Color='green', yerr=error, ecolor='green');
  
    coverage_array_for_alpha_over_multiple_runs = np.load("./DistanceTraversedVSResponseTime_-5.npy")
    mean_coverage_array = np.mean(coverage_array_for_alpha_over_multiple_runs, axis = 0)
    std_coverage_array = np.std(coverage_array_for_alpha_over_multiple_runs,axis = 0)
    error = ((std_coverage_array.T[1] * 1.96) / math.sqrt(no_of_iter));
  

    #Plotting coverage vs number of agents
    '''ax.plot(mean_coverage_array.T[0],mean_coverage_array.T[1], label='Step Penalty = -5',  linestyle='dashed', color='black', linewidth=6)
    ax.errorbar(mean_coverage_array.T[0],mean_coverage_array.T[1], fmt="none", Color='black', yerr=error, ecolor='black');
  

  
    coverage_array_for_alpha_over_multiple_runs = np.load("./DistanceTraversedVSResponseTime_-10.npy")
    mean_coverage_array = np.mean(coverage_array_for_alpha_over_multiple_runs, axis = 0)
    std_coverage_array = np.std(coverage_array_for_alpha_over_multiple_runs,axis = 0)
    error = ((std_coverage_array.T[1] * 1.96) / math.sqrt(no_of_iter));
  

    #Plotting coverage vs number of agents
    ax.plot(mean_coverage_array.T[0],mean_coverage_array.T[1], label='Step Penalty = -10',  linestyle='-.', color='blue', linewidth=6)
    ax.errorbar(mean_coverage_array.T[0],mean_coverage_array.T[1], fmt="none", Color='blue', yerr=error, ecolor='blue');'''
  

    #plt.title("Distance Traversed vs Response Time", fontsize=16, fontweight='bold')
    plt.xlabel("Distance Traversed", fontsize=14, fontweight='bold')
    plt.ylabel("Response Time", fontsize=14, fontweight='bold')

    plt.legend()
    if not os.path.exists("./Figure/%s/"%(now)):
        os.makedirs("./Figure/%s/"%(now))
    plt.savefig("./Figure/%s/DistanceTraversedVSResponseTime.png"%(now))
    plt.close() 




if __name__ == "__main__": 
    plotDiagrams();
        

        

