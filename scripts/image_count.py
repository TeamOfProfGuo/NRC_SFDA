import os
import pdb
import sys


def list_all_files(rootdir):
    _files = []
    lst = os.listdir(rootdir)
    for i in range(0,len(lst)):
        path = os.path.join(rootdir+'/',lst[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
            if path[-4:] == '.png':
                _files.append(path)
    return _files



if __name__ == '__main__':
    root = f'{sys.argv[1]}'
    lst = list_all_files(root)
    pdb.set_trace()