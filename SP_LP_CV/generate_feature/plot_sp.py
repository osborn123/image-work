import marimo

__generated_with = "0.6.19"
app = marimo.App()


@app.cell
def __():
    # data.sp_dist_raw.int().flatten().cpu().bincount()
    target_data = [1159428,   10556,   86332,  247250,  663302, 1187132, 1389500, 1118348,
             693030,  378066,  204848,  109002,   53372,   22528,    7614,    2202,
                592,     130,      30,       2]
    # dist.int().flatten().cpu().bincount()
    ours_data = [  53068,   69418,  406924,  968002, 1539333, 1676425, 1247660,  707880,
             355186,  182876,   86340,   29816,    8424,    1648,     226,      38]
    return ours_data, target_data


@app.cell
def __():
    import matplotlib.pyplot as plt

    # 给定的数据
    data1 = [1159428, 10556, 86332, 247250, 663302, 1187132, 1389500, 1118348,
             693030, 378066, 204848, 109002, 53372, 22528, 7614, 2202,
             592, 130, 30, 2]

    data2 = [53068, 69418, 406924, 968002, 1539333, 1676425, 1247660, 707880,
             355186, 182876, 86340, 29816, 8424, 1648, 226, 38]

    # X 轴坐标
    x1 = list(range(len(data1)))
    x2 = list(range(len(data2)))

    # 创建图形
    plt.figure(figsize=(10, 6))
    plt.plot(x1, data1, label='target distribution', marker='o')  # 使用圆圈标记每个数据点
    plt.plot(x2, data2, label='learned distribution', marker='x', linestyle='--')  # 使用 x 标记每个数据点

    # 设置 x 轴刻度为最长数据集的索引，并确保每个索引都显示为整数
    plt.xticks(range(max(len(data1), len(data2))))

    # 添加标题和图例
    plt.title('Unnormalized Distributions')
    plt.xlabel('Distance number')
    plt.ylabel('Frequency')
    plt.legend()

    # 显示图形
    plt.show()
    return data1, data2, plt, x1, x2


@app.cell
def __():
    import seaborn as sns
    import numpy as np, os
    ours_dist = np.load("ours_dist.npy")
    target_dist = np.load("target_dist.npy")
    return np, os, ours_dist, sns, target_dist


@app.cell
def __():
    # from scipy.stats import gaussian_kde
    # # 创建 KDE 对象
    # kde_ours = gaussian_kde(ours_dist)
    # kde_target = gaussian_kde(target_dist)

    # # 定义值域为 0 到 1，这里使用 1000 个点来计算 CDF
    # x = np.linspace(0, 1, 1000)
    # cdf_ours = kde_ours.integrate_box_1d(0, x)
    # cdf_target = kde_target.integrate_box_1d(0, x)
    return


@app.cell
def __(np, ours_dist, plt, target_dist):
    # import numpy as np
    # import matplotlib.pyplot as plt

    # # 加载数据
    # ours_dist = np.load("ours_dist.npy").flatten()
    # target_dist = np.load("target_dist.npy").flatten()

    # 分箱并计算直方图
    bins = np.linspace(0, 1, 100)  # 定义 100 个等宽的箱
    hist_ours, bin_edges_ours = np.histogram(ours_dist, bins=bins)
    hist_target, bin_edges_target = np.histogram(target_dist, bins=bins)

    # 计算直方图的 x 轴坐标（每个 bin 的中点）
    x_ours = 0.5 * (bin_edges_ours[1:] + bin_edges_ours[:-1])
    x_target = 0.5 * (bin_edges_target[1:] + bin_edges_target[:-1])

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.bar(x_ours, hist_ours, width=np.diff(bin_edges_ours), alpha=0.5, label='Our Distribution', align='center')
    plt.bar(x_target, hist_target, width=np.diff(bin_edges_target), alpha=0.5, label='Target Distribution', align='center')
    plt.title('Normalize Distribution')
    plt.xlabel('Normalize Distribution Number')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    return (
        bin_edges_ours,
        bin_edges_target,
        bins,
        hist_ours,
        hist_target,
        x_ours,
        x_target,
    )


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
