import numpy as np
import matplotlib.pyplot as plt
import pickle
from opt import *
import folsom

np.random.seed(13)

algorithm = PTreeOpt(folsom.f, 
                    feature_bounds = [[0,1000], [1,365], [0,300]],
                    feature_names = ['Storage', 'Day', 'TDI'],
                    discrete_actions = True,
                    action_names = ['Release_Demand', 'Hedge_80', 'Hedge_50', 'Flood_Control'],
                    mu = 10,
                    cx_prob = 0.70,
                    population_size = 50,
                    max_depth = 3
                    )


# snapshots = algorithm.run(max_nfe = 3500, log_frequency = 50)
# pickle.dump(snapshots, open('snapshots.pkl', 'wb'))



snapshots = pickle.load(open('snapshots-reoptimized-13.pkl', 'rb'))

# if animating ...
# for i,P in enumerate(snapshots['best_P']):
#   print str(P)
  # nfestring = 'nfe-' + '%06d' % snapshots['nfe'][i] + '.png'
  # P.graphviz_export('figs/anim/tree-' + nfestring, dpi=150)
  # df = folsom.f(P, mode='simulation')
  # folsom.plot_results(df, filename='figs/anim/folsom-' + nfestring, dpi=150)

  # plt.rcParams['figure.figsize'] = (5, 5)
  # plt.plot(snapshots['nfe'][:i+1], snapshots['best_f'][:i+1], linewidth=2, color='steelblue')
  # plt.xlim([0,np.max(snapshots['nfe'])])
  # plt.ylim([np.min(snapshots['best_f']), np.max(snapshots['best_f'])])
  # plt.ylabel('RMSE')
  # plt.xlabel('NFE')
  # plt.savefig('figs/anim/convergence-' + nfestring, dpi=150)

# else just save the last one with DPI 300, or better yet as SVG


P = snapshots['best_P'][-1]
df = folsom.f(P, mode='simulation')
folsom.plot_results(df)



# algorithm.best_P.graphviz_export('figs/bestPfol.png')

# results = folsom.f(algorithm.best_P, mode='simulation')
# folsom.plot_results(results)

# TEST ONE
# L = [['Flood_Control']]
# L = [[1,256], ['Flood_Control'], ['Release_Demand']]
# P = PTree(L)
# # P.graphviz_export('graphviz/whatever.png')
# results = folsom.f(P, mode='simulation')
# folsom.plot_results(results)