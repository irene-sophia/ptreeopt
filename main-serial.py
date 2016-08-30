import numpy as np
import matplotlib.pyplot as plt
import pickle
from opt import *
from folsom import Folsom
import pandas as pd

np.random.seed(9)

model = Folsom('folsom-daily.csv', sd='1995-10-01', ed='2015-09-30', fit_historical = True)

algorithm = PTreeOpt(model.f, 
                    feature_bounds = [[0,1000], [1,365]],# [0,300]],
                    feature_names = ['Storage', 'Day'],# 'Inflow'],
                    discrete_actions = True,
                    action_names = ['Release_Demand', 'Hedge_90', 'Hedge_80', 'Hedge_70', 'Hedge_60', 'Hedge_50', 'Flood_Control'],
                    mu = 10,
                    cx_prob = 0.70,
                    population_size = 50,
                    max_depth = 3
                    )


snapshots = algorithm.run(max_nfe = 50000, log_frequency = 50)
pickle.dump(snapshots, open('snapshots-fit-hist.pkl', 'wb'))
