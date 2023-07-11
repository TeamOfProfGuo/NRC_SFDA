"""
Usage:
    python scripts/grep_results.py [date] [ignore_args]
    => Show the exp results on that day
"""

import os
import sys
import numpy as np

exp_dict = {} 
exp_list = []

dir_list = os.listdir(f"./result/home/{sys.argv[1]}")

for dir in sorted(dir_list):
    exp_id = f'{sys.argv[1]}_{dir}' 
    log_path = os.path.join(f'./result/home/{sys.argv[1]}', dir, 'log.txt')
    with open(log_path, 'r') as f:
        lines = f.read().split('\n')
    lines = [line for line in lines if line[:6] == '-'*6]
    
    line = lines[-1]
    accs = line[6:].split(',')
    accs = [e.split('=')[1].replace('%', '') for e in accs]
    accs = [float(e) for e in accs]
    max_acc = max(accs)
    
    exp_dict[exp_id] = max_acc
    exp_list.append(max_acc)
    
    print(exp_id, line)
print(exp_dict.values())
print(np.mean(np.array(exp_list)))
    