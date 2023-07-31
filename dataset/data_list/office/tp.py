
dname = 'webcam'
old_path = 'dataset/data_list/office/' + '{}_list.txt'.format(dname)
new_path = 'dataset/data_list/office/' + '{}_list1.txt'.format(dname)

lines = open(old_path).readlines()

with open(new_path, 'w') as f:
    for line in lines:
        new_line = line[22:]
        f.write(new_line)
