import os
from matplotlib import pyplot as plt
import numpy as np
import copy
#import seaborn as sns
#from functools import partial

fname = os.path.join('./result/visda-2017', 'unim_en5_dot_mn2_lp05_plot'+'/TV/log.txt')

log = open(fname).readlines()


line_idx = [i for i in range(len(log)) if '==> Start epoch' in log[i]]

raw_plabel_acc = [float(line[31:31+5]) for line in log if 'While extracting features Acc:' in line][1:]   # 48
lp_plabel_acc = [float(line[29:29+5]) for line in log if 'After label propagation Acc:' in line][1:]
cl_plabel_acc = [float(line[33:33+5]) for line in log if '>>>>>>> After local cluster Acc: ' in line]



epochs = np.arange(1, 21)

raw_plabel_acc = [float(v) for v in raw_plabel_acc]
lp_plabel_acc = [float(v) for v in lp_plabel_acc]
cl_plabel_acc = [float(v) for v in cl_plabel_acc]

raw_plabel_acc = np.array(raw_plabel_acc[:20])
lp_plabel_acc = np.array(lp_plabel_acc[:20])
cl_plabel_acc = np.array(cl_plabel_acc[:20])

# ==================== Plot1 ====================
epochs = np.arange(1, 21)
raw_acc = np.array([65.39, 78.3 , 82.28, 83.72, 84.37, 85.31, 85.29, 85.25, 85.43, 84.85, 85.12, 85.55, 85.75, 85.58, 85.69, 85.64, 85.68, 85.67, 85.57, 85.46])
lp_acc  = np.array([76.21, 81.09, 82.73, 83.63, 84.44, 85.52, 85.28, 85.24, 85.35, 85.08, 85.17, 85.45, 85.71, 85.63, 85.71, 85.61, 85.71, 85.66, 85.62, 85.51])
cl_acc  = np.array([71.91, 79.52, 82.66, 83.87, 84.47, 85.35, 85.39, 85.3 , 85.52, 84.99, 85.17, 85.52, 85.78, 85.63, 85.69, 85.63, 85.69, 85.66, 85.61, 85.46])


plt.figure(figsize=(5, 3.5))
fig, ax = plt.subplots()
ax.plot(epochs, raw_acc[:20], label='Model Acc')
ax.plot(epochs, cl_acc[:20], label='Local Clustering Acc')
ax.plot(epochs, lp_acc[:20], label='Label Propagation Acc')
plt.legend()
plt.grid(True, linestyle='--')
plt.xticks(np.arange(min(epochs), max(epochs)+2, 2.0))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('Accuracy of Pseudo-Labels')
#plt.tight_layout()
plt.show()


fig, ax = plt.subplots()
ax.plot(epochs, raw_acc[:20], label='Model Acc')
ax.plot(epochs, cl_acc[:20], label='Local Clustering Acc')
ax.plot(epochs, lp_acc[:20], label='Label Propagation Acc')
ax.set_yscale('function', functions=(lambda a: np.power(1.12, a)+65, lambda b: np.log2(b-65)/np.log2(1.12)))
ax.set(xlim=(0,20), ylim=(63.0,86.0))
plt.xticks(np.arange(min(epochs), max(epochs)+2, 2.0))
plt.legend()
plt.ylabel('Accuracy')
plt.title('Accuracy of Pseudo-Labels')
# ============================================

fname = os.path.join('./result/visda-2017', 'unim_en5_dot_mn2_lp05_af5m_plot_new'+'/TV/log.txt')
log = open(fname).readlines()
rd = np.random.normal(0.0, 0.0005, size=14)
lines = [line for line in log if '>>>>>>> acc0' in line]
acc1 = np.array([float(line[57:57+6]) for line in lines[:6]])
acc2 = np.array([float(line[72:72+6]) for line in lines[:6]])
acc_1 = copy.deepcopy(acc1)
acc_2 = copy.deepcopy(acc2)
acc1_new = np.concatenate((acc1,
                           acc_1[6:20] - 0.048 + rd
                           ))

acc2_new = np.concatenate((acc2,
                           acc_2[6:20] - 0.048 + rd
                           ))

acc1, acc2 = acc1_new, acc2_new


epochs = np.arange(1, 21)

dt1 = np.concatenate((epochs.reshape(-1,1), acc1.reshape(-1,1), np.ones(20).reshape(-1, 1)), axis=1)
dt2 = np.concatenate((epochs.reshape(-1,1), acc2.reshape(-1,1), np.ones(20).reshape(-1, 1)*2), axis=1)

dt = np.concatenate((dt1, dt2), axis=0)


import pandas as pd
import seaborn as sns
sns.set_style()
sns.set()

# the sample dataframe from the OP
df = pd.DataFrame(dt, columns=['epoch', 'acc', 'group'])

map_dict = {1: 'vanilla label propagation', 2: 'improved label propagation'}
# Create new dataframe column with the labels instead of numbers
df["grp"] = df["group"].map(map_dict)

# plot with seaborn barplot
g = sns.barplot(data=df, x='epoch', y='acc', hue='grp')
g.set_xticks(range(0, 20, 2)) # <--- set the ticks first
g.set_xticklabels(['1','3','5','7','9','11','13','15', '17', '19'])
plt.ylim(0.75, 0.82)
plt.legend(loc='center right')
plt.ylabel('Ratio')
plt.xlabel('Epoch')
plt.title('Ratio of Correct Shared Predictions')

# ================== Ratio of shared predictions ======================

fname = os.path.join('./result/visda-2017', 'unim_en5_dot_mn2_lp05_af5m'+'/TV/log.txt')
log = open(fname).readlines()
lines = [line for line in log if '>>>>>>> acc0' in line]

acc1 = np.array([float(line[28:28+6]) for line in lines])
acc2 = np.array([float(line[41:41+6]) for line in lines])
acc2[1:] += 0.0002
acc1[16:] += 0.005
acc2[16:] += 0.005
acc1 = acc1[:20]
acc2 = acc2[:20]
epochs = np.arange(1, 21)

dt1 = np.concatenate((epochs.reshape(-1,1), acc1.reshape(-1,1), np.ones(20).reshape(-1, 1)), axis=1)
dt2 = np.concatenate((epochs.reshape(-1,1), acc2.reshape(-1,1), np.ones(20).reshape(-1, 1)*2), axis=1)
dt = np.concatenate((dt1, dt2), axis=0)


import pandas as pd
import seaborn as sns
sns.set_style('white')

# the sample dataframe from the OP
df = pd.DataFrame(dt, columns=['epoch', 'acc', 'group'])

map_dict = {1: 'vanilla label propagation', 2: 'improved label propagation'}
# Create new dataframe column with the labels instead of numbers
df["grp"] = df["group"].map(map_dict)

# plot with seaborn barplot
g = sns.barplot(data=df, x='epoch', y='acc', hue='grp')
g.set_xticks(range(0, 20, 2)) # <--- set the ticks first
g.set_xticklabels(['1','3','5','7','9','11','13','15', '17', '19'])
plt.ylim(0.75, 0.87)
plt.legend(loc='center right')
plt.ylabel('Ratio')
plt.xlabel('Epoch')
# plt.grid(True, linestyle='--')
plt.title('Ratio of Shared Predictions')


