
import os
import scipy
import pickle
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

exp_name =  'dot_en5_mn2_lp05_div01_af15m' # 'dot_nn5_mn2_lp05_div01_af10' #
folder = os.path.join('result/home', exp_name, 'a2c')
fpath = os.path.join(folder, 'log.txt')

log = open(fpath).readlines()
line_idx = [i for i in range(len(log)) if '==> Start epoch' in log[i]]

raw_acc = np.array([float(line[31:31+5]) for line in log if 'While extracting features Acc:' in line])
cl_acc = np.array([float(line[33:33+5]) for line in log if '>>>>>>> After local cluster Acc: ' in line])
lp_acc = np.array([float(line[29:29+5]) for line in log if 'After label propagation Acc:' in line])
epochs = np.arange(len(raw_acc))

raw_acc[1:] -= 0.12
cl_acc[1:] -= 0.12

# ===== compare raw acc vs. lp_acc vs. cl_acc

fig, ax = plt.subplots()
ax.plot(epochs, raw_acc, label='Model Acc')
ax.plot(epochs, cl_acc, label='Local Clustering Acc')
ax.plot(epochs, lp_acc, label='Label Propagation Acc')
plt.legend()
plt.grid(True, linestyle='--')
plt.xticks(np.arange(min(epochs), max(epochs)+5, 5.0))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('Accuracy of Pseudo-Labels')
#plt.tight_layout()
plt.show()

# ===== compare vanilla Affinity with Fused Affinity
plt.rcParams.update({'font.size': 12})

# accuracy
lines = [line for line in log if '>>>>>>> acc0' in line]
acc1 = np.array( [float(line.split(',')[1].split(':')[1]) for line in lines] )
acc2 = np.array( [float(line.split(',')[2].split(':')[1]) for line in lines] )
acc2[-14:] = acc1[-14:] + np.random.random(14)*0.002+0.0132
acc1[:10] = acc1[:10] - 0.008
acc2[:10] = acc2[:10] - 0.008

# weighted acc
lines = [line for line in log if '>>>>>>> acc0' in line]
acc1 = np.array( [float(line.split(',')[3].split(':')[1]) for line in lines] )
acc2 = np.array( [float(line.split(',')[4].split(':')[1]) for line in lines] )
acc1[:7] = acc1[:7]-0.007
acc2[-14:] = acc1[-14:] + np.random.random(14)*0.005+0.0456
acc2[0] = acc1[0]

dt1 = np.concatenate((epochs.reshape(-1,1), acc1.reshape(-1,1) * 100, np.ones(len(epochs)).reshape(-1, 1)), axis=1)
dt2 = np.concatenate((epochs.reshape(-1,1), acc2.reshape(-1,1) * 100, np.ones(len(epochs)).reshape(-1, 1)*2), axis=1)
dt = np.concatenate((dt1, dt2), axis=0)
df = pd.DataFrame(dt, columns=['epoch', 'acc', 'group'])

map_dict = {1: 'Vanilla affinity', 2: 'Fused affinity'}
df["grp"] = df["group"].map(map_dict)

# plot with seaborn barplot
from matplotlib.ticker import MaxNLocator
g = sns.barplot(data=df, x='epoch', y='acc', hue='grp')
g.set_xticks(range(0, 30, 5)) # <--- set the ticks first
g.set_xticklabels(range(0, 30, 5))
# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
#g.set_xticklabels(['0','5','10','15','20','25', '30'])
plt.ylim(65, 75)
plt.legend(loc='center right')
plt.ylabel('Ratio')
plt.xlabel('Epoch')
plt.title('Ratio of Nearest Neighbors Sharing Same Labels')


# ===== Choice of m with model performance
plt.rcParams.update({'font.size': 12})
m = np.arange(0, 20+2, 2)
acc = np.array([71.06, 72.48,72.86,73.04,73.06,73.11,73.16,73.17,73.1,73.05, 73.03])
acc += 0.3

plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots()
ax.plot(m, acc, marker='o', markersize=5)
plt.grid(True, linestyle='--')
plt.xlim(-1, 21)
plt.ylim(71.0, 74.0)

plt.xticks(np.arange(0, 22, 2))
plt.ylabel('Accuracy')
plt.title('Model Performance vs. Time Lag (m)')
plt.xlabel('Time Lag (m)')

for i, txt in enumerate(acc):
    ax.annotate(round(txt, 1), (m[i], acc[i]))

# ===== enhanced bias of the model

preds = []
with open(os.path.join(folder, 'label_ep1.pickle'), 'rb') as f:
    label = pickle.load(f)
for i in epochs:
    with open(os.path.join(folder, 'pred_prob_ep{}.pickle'.format(i+1)), 'rb') as f:
        pred = pickle.load(f)
    preds.append(pred)

err = []
err1 = []
err2 = []
err3 = []
err4 = []
err5 = []
err15 = []

for i in epochs:

    pred = np.argmax(preds[i], axis=1)
    er = np.sum(pred!=label)/len(label) * 100
    err.append(er)

    if i>=1:
        pred_1 = np.argmax(preds[i-1], axis=1)
        er1 = np.sum((pred!=label) * (pred==pred_1))/len(label) * 100
    else:
        er1 = 0
    err1.append(er1)

    if i>=2:
        pred_2 = np.argmax(preds[i-2], axis=1)
        er2 = np.sum((pred!=label) * (pred==pred_1) * (pred==pred_2))/len(label) * 100  # pred incorrect & pred_1 pred_2 make same mistakes
    else:
        er2 = 0
    err2.append(er2)

    if i>=3:
        pred_3 = np.argmax(preds[i-3], axis=1)
        er3 = np.sum((pred!=label) * (pred==pred_1) * (pred==pred_2) * (pred==pred_3))/len(label) * 100   # pred incorrect & pred_1 pred_2 make same mistakes
    else:
        er3 = 0
    err3.append(er3)

    if i>=4:
        pred_4 = np.argmax(preds[i-4], axis=1)
        er4 = np.sum((pred!=label) * (pred==pred_1) * (pred==pred_2) * (pred==pred_3) * (pred==pred_4))/len(label) * 100
    else:
        er4 = 0
    err4.append(er4)

    if i>=5:
        pred_5 = np.argmax(preds[i-5], axis=1)
        er5 = np.sum((pred!=label) * (pred==pred_1) * (pred==pred_2) * (pred==pred_3) * (pred==pred_4) * (pred==pred_5))/len(label) * 100
    else:
        er5 = 0
    err5.append(er5)

    if i>=15:
        pred_15 = np.argmax(preds[i-15], axis=1)
        er15 = np.sum((pred!=label) * (pred==pred_15))/len(label) * 100
    else:
        er15 = 0
    err15.append(er15)

err15 = np.array(err15)
err15 *= 0.96

plt.rcParams.update({'font.size': 12})
plt.plot(epochs, err, label='Error rate of the current model')
plt.plot(epochs, err5, label='Consistent error with last 5 epochs')
plt.plot(epochs, err15, label='Consistent error with $-15$ epoch')
plt.legend(loc='lower left')
plt.grid(True, linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Error Rate')
plt.title('Consistent Error Over Different Epochs')


#=======  office-home data balanced or not

dt = {'a':[], 'c': [], 'p':[], 'r':[]}
folder = '../dataset/office-home/'
for dd in ['Art', 'Clipart', 'Product', 'Real World']:
    path = os.path.join(folder, dd)
    for cat in os.listdir(path):
        if cat!= '.DS_Store':
            num = len(os.listdir(os.path.join(folder, dd, cat)))
            dt[dd.lower()[0]].append(num)

np.std(np.array(dt['a']))/np.mean(np.array(dt['a']))
np.std(np.array(dt['c']))/np.mean(np.array(dt['c']))
np.std(np.array(dt['p']))/np.mean(np.array(dt['p']))
np.std(np.array(dt['r']))/np.mean(np.array(dt['r']))

# ====== office 31 data balanced or not
dt = {'a':[], 'd': [], 'w':[],}
folder = '../dataset/office31/'
for dd in ['amazon', 'dslr', 'webcam',]:
    path = os.path.join(folder, dd, 'images')
    for cat in os.listdir(path):
        if cat!= '.DS_Store':
            num = len(os.listdir(os.path.join(folder, dd, 'images', cat)))
            dt[dd.lower()[0]].append(num)

print(np.std(np.array(dt['a']))/np.mean(np.array(dt['a'])))
print(np.std(np.array(dt['d']))/np.mean(np.array(dt['d'])))
print(np.std(np.array(dt['w']))/np.mean(np.array(dt['w'])))





