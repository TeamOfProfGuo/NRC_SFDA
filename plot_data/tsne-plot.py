import pdb
import sys
import torch
import random
import numpy as np

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap


def select_samples(cls_number, feats, labels, num_samples):
    for i in range(cls_number):
        # get the current cls index
        cur_cls_idx = torch.where(labels == i)[0]

        # shuffle the index to randomly select samples for each cls 
        shuffled_indices = torch.randperm(len(cur_cls_idx))

        selected_idx = cur_cls_idx[shuffled_indices[:num_samples]]

        new_feats = feats[selected_idx]
        new_labels = labels[selected_idx]

        if i == 0:
            plot_feats = new_feats
            plot_label = new_labels
        else:
            plot_feats = torch.cat((plot_feats, new_feats), dim=0)
            plot_label = torch.cat((plot_label, new_labels), dim=0)
    
    return plot_feats.numpy(), plot_label.numpy()


def tsne_plot(cls_number, feats, labels, num_samples, title, file_name):
    plot_feats, plot_label= select_samples(cls_number, feats, labels, num_samples)

    print("Done with selecting samples")

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(plot_feats)
    print("Done with fitting data")

    np.save(f"plot_data{num_samples}_{file_name}.npy", tsne_result)
    np.save(f"plot_label{num_samples}_{file_name}.npy", plot_label)

    # Create a list of distinct colors for plotting
    distinct_colors = plt.cm.get_cmap("tab20", cls_number)

    print("Start plotting")
    # Plot the t-SNE graph with different colors for each class
    plt.figure(figsize=(40, 32))
    legend_patches = []
    for class_label in range(cls_number):
        class_indices = np.where(plot_label == class_label)[0]
        plt.scatter(
            tsne_result[class_indices, 0],
            tsne_result[class_indices, 1],
            c=[distinct_colors(class_label)],
            label=f"Class {class_label}",
            alpha=1.0,
            s=130
        )
        legend_patch = mpatches.Patch(color=distinct_colors(class_label), label=f"Class {class_label}")
        legend_patches.append(legend_patch)

    print(f"Done with {file_name}")

    plt.title(title, fontsize=100, pad=110)
    plt.xticks([])  # Hide x-axis tick labels and labels
    plt.yticks([])  # Hide y-axis tick labels and labels

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
    

if __name__ == "__main__":
    num_samples = sys.argv[1]
    num_lists = [int(num_samples[i:i+3]) for i in range(0, len(num_samples), 3)]

    # set the seed
    SEED = int(sys.argv[2])
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    # Load the tensor from the specified file
    all_feats_before = torch.load("all_feats_before.pt")
    all_labels_before = torch.load("all_labels_before.pt")

    for num in num_lists:
        tsne_plot(12, all_feats_before, all_labels_before, num_samples=num, title="Before Batchnorm Adapt", file_name="before")

    # clear memory
    del all_feats_before, all_labels_before

    all_feats_after = torch.load("all_feats_after.pt")
    all_labels_after = torch.load("all_labels_after.pt")

    for num in num_lists:
        tsne_plot(12, all_feats_after, all_labels_after, num_samples=num, title="After Batchnorm Adapt", file_name="after")

