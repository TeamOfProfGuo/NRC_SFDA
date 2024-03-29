

import numpy as np
from matplotlib import pyplot as plt

acc = None
acc1 = None

inv_cmap = plt.get_cmap('RdBu').reversed()
im = plt.imshow(acc, cmap=inv_cmap)
plt.colorbar(im)

inv_cmap = plt.get_cmap('RdBu').reversed()
im = plt.imshow(acc1, cmap=inv_cmap)
plt.colorbar(im)