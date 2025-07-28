# !! Under Construction
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from matplotlib.lines import Line2D


__all__ = [
    "plot_curve",
    "plot_contrast_curves",
    "plot_curves",
    "plot_curves_demonstrate",
    "plot_funcs",
    "plot_funcs_demonstrate",
    "plot_plane_scatter",
    "plot_line_scatter",
    "clear_plots",
]


# ----------------------------------------------------------------------------------------
# 这个函数绘制曲线并保存图像
def plot_curve(x_data, y_data, curve_label, x_label, y_label, pic_title, pic_path):
    plt.figure(figsize=(8, 4))
    plt.title(pic_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    plt.plot(x_data, y_data, label=curve_label)
    
    plt.legend()
    plt.grid()
    plt.savefig(pic_path)
    plt.close()

# ----------------------------------------------------------------------------------------
# 这个函数绘制曲线关于背景曲线的对比图并保存图像
def plot_contrast_curves(x_data, y_data, background_y_data, curve_label, background_curve_label, x_label, y_label, pic_title, pic_path, background_x_data = None):
    if background_x_data == None: background_x_data = x_data
    plt.figure(figsize=(8, 6))
    plt.plot(x_data, y_data, label=curve_label)
    plt.plot(background_x_data, background_y_data, label=background_curve_label, linestyle='--')
    plt.title(pic_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid()
    plt.savefig(pic_path)
    plt.close()

# ----------------------------------------------------------------------------------------
# 这个函数绘制多条曲线并保存图像
def plot_curves(x_data, y_datas, curve_labels, x_label, y_label, pic_title, pic_path, ylim = None):
    plt.figure(figsize=(8, 4))
    plt.title(pic_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    assert len(y_datas) == len(curve_labels)
    for i in range(len(y_datas)):
        plt.plot(x_data, y_datas[i], label=curve_labels[i])
    if ylim != None:
        plt.ylim(ylim[0], ylim[1])
    plt.legend()
    plt.grid()
    plt.savefig(pic_path)
    plt.close()
    
# ----------------------------------------------------------------------------------------
# 这个函数绘制多条曲线并显示图像
def plot_curves_demonstrate(x_data, y_datas, curve_labels, x_label, y_label, pic_title, ylim = None):
    plt.figure(figsize=(8, 4))
    plt.title(pic_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    assert len(y_datas) == len(curve_labels)
    for i in range(len(y_datas)):
        plt.plot(x_data, y_datas[i], label=curve_labels[i])
    if ylim != None:
        plt.ylim(ylim[0], ylim[1])
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()
    
# ----------------------------------------------------------------------------------------
# 这个函数绘制多个函数并保存图像
def plot_funcs(funcs, plot_range, point_num, curve_labels, x_label, y_label, pic_title, pic_path, ylim = None):
    x_data = np.linspace(plot_range[0], plot_range[1], point_num)
    y_datas = []
    for func in funcs:
        y_data = []
        for x in x_data:
            y_data.append(func(x))
        y_datas.append(y_data)
    plot_curves(x_data, y_datas, curve_labels, x_label, y_label, pic_title, pic_path, ylim)
    
# ----------------------------------------------------------------------------------------
# 这个函数绘制多个函数并显示图像
def plot_funcs_demonstrate(funcs, plot_range, point_num, curve_labels, x_label, y_label, pic_title, ylim = None):
    x_data = np.linspace(plot_range[0], plot_range[1], point_num)
    y_datas = []
    for func in funcs:
        y_data = []
        for x in x_data:
            y_data.append(func(x))
        y_datas.append(y_data)
    plot_curves_demonstrate(x_data, y_datas, curve_labels, x_label, y_label, pic_title, ylim)
    
def plot_plane_scatter(
    ndarray_path, 
    pic_path, 
    pic_title = "2D Scatter Plot", 
    x_label = "X-axis", 
    y_label = "Y-axis", 
    point_size = 50, 
    color_path = None, 
    color_labels = None, 
    markersize = None
)-> None:
    
    data = np.load(ndarray_path)
    if data.shape[1] != 2:
        raise ValueError("The second dimension of the array should be 2 (x and y coordinates).")
    
    points = data.T
    
    if color_path is not None and color_labels is not None:
        colors = np.load(color_path).flatten()
        if len(colors) != data.shape[0] or len(color_labels) < np.max(colors) + 1:
            raise ValueError("The length of 'colors' should be equal to the number of points, and 'color_labels' should be no less than max(colors)+1.")
        unique_colors = list(range(len(color_labels)))
        color_map = plt.cm.get_cmap('viridis', len(unique_colors) + 1)
        normalized_colors = np.array(colors + 1, dtype=float) / (len(unique_colors) + 1 - 1)
        colors_for_scatter = color_map(normalized_colors)
        legend_handles = []
        for color_value, label in zip(unique_colors, color_labels):
            color_rgb: Tuple[float, float, float] = tuple(color_map(color_value / (len(unique_colors) - 1))) # type: ignore
            if markersize is None:
                patch = Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color_rgb, markersize=point_size)
            else:
                patch = Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color_rgb, markersize=markersize)
            legend_handles.append(patch)
        
        plt.scatter(points[0], points[1], s=point_size, c=colors_for_scatter)
        plt.legend(handles=legend_handles, loc='best')
    else:
        plt.scatter(points[0], points[1], s=point_size)
    
    plt.title(pic_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    if not os.path.exists(os.path.dirname(pic_path)):
        os.makedirs(os.path.dirname(pic_path))
    plt.savefig(pic_path)
    plt.close()
    
def plot_line_scatter(ndarray_path, pic_path, pic_title="1D Scatter Plot", x_label="Index", y_label="Value", point_size=50):
    x = np.load(ndarray_path).flatten()
    y = np.random.randn(*x.shape) * 0.1

    plt.scatter(x, y, s=point_size)
    plt.title(pic_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if not os.path.exists(os.path.dirname(pic_path)):
        os.makedirs(os.path.dirname(pic_path))
    
    plt.savefig(pic_path)
    plt.close()
    
# ----------------------------------------------------------------------------------------
# 这个函数清除指定路径下所有后缀为.png的文件
def clear_plots(pic_path):
    pattern = os.path.join(pic_path, '*.png')
    png_files = glob.glob(pattern)
    for file_path in png_files:
        try:
            os.remove(file_path)
            print(f"已删除文件: {file_path}")
        except Exception as e:
            print(f"删除文件 {file_path} 时出错: {e}")