import pdb
import sys
import torch
import random
import os
import numpy as np

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
folder = 'plot_data'

cls_number=12
num_samples=450
title="After Batchnorm Adaptation"
file_name="after"

tsne_result = np.load(os.path.join(folder, f"plot_data450_{file_name}.npy"))
plot_label =  np.load(os.path.join(folder, f"plot_label450_{file_name}.npy"))
print("Done with loading data")

# Create a list of distinct colors for plotting
distinct_colors = plt.cm.get_cmap("tab20", cls_number)

print("Start plotting")
# Plot the t-SNE graph with different colors for each class
legend_patches = []
for class_label in range(cls_number):
    class_indices = np.where(plot_label == class_label)[0]
    plt.scatter(
        tsne_result[class_indices, 0],
        tsne_result[class_indices, 1],
        c=[distinct_colors(class_label)],
        label=f"Class {class_label}",
        alpha=1.0,
        s=2
    )
    legend_patch = mpatches.Patch(color=distinct_colors(class_label), label=f"Class {class_label}")
    legend_patches.append(legend_patch)
# plt.xlim(-50, 50)
plt.suptitle(title, fontsize=100, pad=110)
plt.xticks([])  # Hide x-axis tick labels and labels
plt.yticks([])  # Hide y-axis tick labels and labels


print(f"Done with {file_name}")

# Create a custom legend using legend_patches
legend = plt.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=6, markerscale=2, prop={"size": 37}
)

plt.subplots_adjust(bottom=0.2)

if file_name == "before":
    # Set x-axis limits
    x_min = -52  # Minimum x-axis value
    x_max = 47  # Maximum x-axis value
    plt.xlim(x_min, x_max)

for text in legend.get_texts():
    text.set_fontsize(45)  # Increase font size for legend text

plt.savefig(f"plot/{num_samples}_{file_name}.png")



