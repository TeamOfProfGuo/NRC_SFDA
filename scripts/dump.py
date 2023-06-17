
import os
import pdb
import sys

# SEP_INTERVAL = 4

parse_metric = lambda x: x.strip().split()[-1]
parse_pretrain_eps = lambda x: list(map(parse_metric, [x[6], x[3], x[4]]))
parse_pretrain_std = lambda x: list(map(parse_metric, [x[7], x[4], x[5]]))
pad_string = lambda x, l: [i.ljust(j) for i, j in zip(x, l)]

dump = []
root = f'{sys.argv[1]}'
column_size = [50, 30, 30, 30]
column_name = ['exp_id', 'label prop mean acc', 'finetune mean acc', 'ft > labelprop']

# def get_file_name(path):
#     f_list = os.listdir(path)
#     files = []
#     for i in f_list:
#         if i[-4:] == '.log':
#             files.append(i)
#     return files

def list_all_files(rootdir):
    _files = []
    lst = os.listdir(rootdir)
    for i in range(0,len(lst)):
        path = os.path.join(rootdir+'/',lst[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
            if path[-4:] == '.log':
                _files.append(path)
    return _files

print(f'\ndumping results under {root}:\n')
print(' '.join(pad_string(column_name, column_size)))

file_lst = list_all_files(root)

for i, d in enumerate(sorted(file_lst)):
    path = d
    with open(path, 'r') as f:
        after_label = []
        max_after_label = 0.0
        max_after_label_idx = 0
        
        after_ft = []
        max_after_ft = 0.0
        max_after_ft_idx = 0

        exp_id = None
        for l in f.read().split('\n'):
            # deal with exp_id
            if l[:8] == '[exp_id]':
                exp_id = l[10:]
            # deal with Epoch Results
            if 'After label' in l:
                current_macc = float(l.split(" ")[-1][:-1])
                after_label.append(current_macc)

                if current_macc > max_after_label:
                    max_after_label = current_macc
                    max_after_label_idx = len(after_label) - 1

            if 'After fine-tuning' in l:
                current_macc = float(l.split(" ")[-3][:-1])
                after_ft.append(current_macc)

                if current_macc > max_after_ft:
                    max_after_ft = current_macc
                    max_after_ft_idx = len(after_ft) - 1

    if len(after_label) == 0 or exp_id is None or len(after_ft) == 0:
        continue
    else:
        # only want macc > 86 ones
        if max_after_label > 86 or max_after_label > 86:
            if max_after_ft > max_after_label:
                try:
                    results = [exp_id, str(after_label[max_after_ft_idx]), str(max_after_ft), 'True']
                except:
                    results = [exp_id, str(after_label[max_after_ft_idx-1]), str(max_after_ft), 'True']

            else:
                try:
                    results = [exp_id, str(max_after_label), str(after_ft[max_after_label_idx]), 'False']
                except:
                    results = [exp_id, str(max_after_label), str(after_ft[max_after_label_idx-1]), 'False']

    # print results
    print(' '.join(pad_string(results, column_size)))
        