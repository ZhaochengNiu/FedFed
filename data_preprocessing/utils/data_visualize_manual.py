import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
# 导入了matplotlib、numpy、sys、os、argparse等模块，用于绘图、数学运算、系统交互等。
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from collections import OrderedDict

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
# 将当前目录的上上级目录添加到系统路径，以便能够导入相关模块。

# 这段代码是一个Python脚本，用于生成和显示一个热力图（heatmap），通常用于展示数据的分布情况。
# 热力图在这里被用来展示不同客户端（clients）拥有的不同类别（classes）的数据样本数量。以下是对代码中每个部分的详细解释：
# 整体来看，这段代码提供了一个从命令行参数生成特定数据分布的热力图，并将其显示和保存的功能。
# 代码中使用了matplotlib库来生成热力图，并使用argparse库来解析命令行参数。


def update_fontsize(ax, fontsize=12.):
    # 这个函数接受一个 matplotlib 轴对象 ax 和字体大小 fontsize，更新轴上的标题、标签等的字体大小。
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                             ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)


# Reference: https://matplotlib.org/stable/gallery/statistics/hist.html
def heatmap(data, row_ticks_to_show, col_ticks_to_show, 
            row_labels, col_labels, ax=None, fontsize=15,
            cbar_kw={}, cbarlabel="", **kwargs):
    # 这个函数接受数据矩阵、行列标签、轴对象、字体大小等参数，生成一个热力图，并添加颜色条（colorbar）。
    """
    Create a heatmap from a numpy array and two lists of labels.
    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    update_fontsize(cbar.ax, fontsize)
    # cbar.ax.set_fontsize(fontsize)

    # We want to show all ticks...
    # ax.set_xticks(np.arange(data.shape[1]))
    # ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticks(col_ticks_to_show)
    ax.set_yticks(row_ticks_to_show)
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=0)
    ax.grid(which="minor", color="r", linestyle='-', linewidth=0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_ylabel('Class ID')
    ax.set_xlabel('Party ID')
    update_fontsize(ax, fontsize=fontsize)

    return im, cbar



def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    # 这个函数用于在热力图上添加注释文本，显示数据值。
    """
    A function to annotate a heatmap.
    We're not using this for our graphs
    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--partition_method', type=str, default='LDA', help='Indicate partition method')
    parser.add_argument('--alpha', type=float, default=0.5, help='Partition Alpha')
    parser.add_argument('--dominant_num', type=int, default=100, help='')
    parser.add_argument('--tail_num', type=int, default=1, help='')
    parser.add_argument('--data_dir', type=str, default='/home/chaoyanghe/BruteForce/FedML/data/pascal_voc'
                                                        '/benchmark_RELEASE', help='Dataset directory')
    parser.add_argument('--dataset', type=str, default='pascal_voc', help="Name of dataset")
    parser.add_argument('--client_num_in_total', type=int, default=10,
                        help='Number of total clients')

    args = parser.parse_args()
    # 使用 argparse 库解析命令行参数，获取分区方法、α值、主导类别数量、尾部类别数量等信息。
    alpha = args.alpha
    client_num = args.client_num_in_total
    class_num = 10
    # classes = list(range(1, 20 + 1, 1))

    dominant_num = args.dominant_num
    tail_num = args.tail_num
    classes = list(range(10))
    clients = list(range(client_num))

    # 根据提供的参数，生成一个字典，其中包含每个客户端拥有的各类别的数据样本数量。
    train_data_cls_counts = {}
    for i in clients:
        train_data_cls_counts[i] = {}
        for class_j in classes:
            if class_j == i:
                train_data_cls_counts[i][class_j] = dominant_num
            else:
                train_data_cls_counts[i][class_j] = tail_num

    print(train_data_cls_counts, classes)
    # Adding missing classes to list
    for key in train_data_cls_counts:
        if len(classes) != len(train_data_cls_counts[key]):
            # print(len(classes))
            # print(len(train_data_cls_counts[key]))
            add_classes = set(classes) - set(train_data_cls_counts[key])
            # print(add_classes)
            for e in add_classes:
                train_data_cls_counts[key][e] = 0

    classes = list(train_data_cls_counts[0].keys())
    # clients = list(range(client_num))

    # Sort the class key values to easily convert to array while preserving order
    # 将客户端数据类别统计信息转换为numpy数组，并转置，以便用于生成热力图。
    samples = []
    for key in train_data_cls_counts:
        od = OrderedDict(sorted(train_data_cls_counts[key].items()))
        samples.append(list(od.values()))
    data = np.array(samples)
    transpose_data = data.T

    # 使用matplotlib生成热力图，并使用show函数显示。
    fig, ax = plt.subplots()
    print(transpose_data, classes, clients)
    fontsize = 18
    im, cbar = heatmap(transpose_data, classes, clients, classes, clients, ax=ax,
                       fontsize=fontsize, cmap="YlGn", cbarlabel="samples")
    annotate_heatmap(im, valfmt="{x:d}", size=7, threshold=20,
                    textcolors=("red", "black"))
    fig.set_figheight(5)
    fig.set_figwidth(7)
    fig.tight_layout()
    plt.show()

    plt.savefig('Dom' + str(dominant_num) + '_tail' + str(tail_num) + \
        '_classes' + str(class_num) + '_clients' + str(client_num) + 'manul.pdf')
    # 将生成的热力图保存为PDF文件。

