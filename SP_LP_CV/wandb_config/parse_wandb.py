import marimo

__generated_with = "0.6.19"
app = marimo.App()


@app.cell
def __():
    import pandas as pd 
    import wandb
    import marimo as mo
    import os
    os.environ["WANDB_API_KEY"]="c1452b5bae77778a04b9cad2a8d96d4424088383"
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs("yangbn/multi_source_retrieval")

    summary_list, config_list, name_list = [], [], []
    sweep_list = []
    runs_list = []
    for run in runs: 
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        if run.sweep is None:
            continue
        elif run.sweep.id != "cp4f8wlu":
            continue
        else:
            runs_list.append(run)
    return (
        api,
        config_list,
        mo,
        name_list,
        os,
        pd,
        run,
        runs,
        runs_list,
        summary_list,
        sweep_list,
        wandb,
    )


@app.cell
def __(runs_list):
    print(len(runs_list))
    return


@app.cell
def __():
    return


@app.cell
def __(mo, runs_list):
    option = list(runs_list[0].summary._json_dict.keys()) + list(runs_list[0].config.keys())
    multiselect = mo.ui.multiselect(options=option)
    return multiselect, option


@app.cell
def __(mo, multiselect):
    mo.vstack([multiselect, mo.md(f"Has value: {multiselect.value}")])
    return


@app.cell
def __(multiselect):
    selected_keys = multiselect.value
    return selected_keys,


@app.cell
def __(runs_list, selected_keys):
    get_run_list = []
    for get_run in runs_list:
        get_run_dict = {}
        get_run_summary = get_run.summary._json_dict
        get_run_config = get_run.config
        for key, value in get_run_summary.items():
            if key in selected_keys:
                get_run_dict[key] = value
        for key, value in get_run_config.items():
            if key in selected_keys:
                get_run_dict[key] = value
        # get_config_summary = get
        get_run_list.append(get_run_dict)
    print(len(get_run_list))
    return (
        get_run,
        get_run_config,
        get_run_dict,
        get_run_list,
        get_run_summary,
        key,
        value,
    )


@app.cell
def __(get_run_list, pd):
    # import pandas as pd
    get_df = pd.DataFrame(get_run_list)
    return get_df,


@app.cell
def __(get_run_list):
    get_run_list
    return


@app.cell
def __(get_df):
    get_df_group = get_df.groupby('select_proportion')
    return get_df_group,


@app.cell
def __(get_df):
    get_df.columns
    return


@app.cell
def __(get_df, mo):
    min_keys = mo.ui.multiselect(options=list(get_df.columns))
    return min_keys,


@app.cell
def __(min_keys, mo):
    mo.vstack([min_keys, mo.md(f"Has value: {min_keys.value}")])
    return


@app.cell
def __(min_keys):
    min_dict = {k: "min" for k in min_keys.value}
    return min_dict,


@app.cell
def __(get_df_group, min_dict):
    get_df_group.agg(min_dict).to_csv("error_report.csv")
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __():
    return


@app.cell
def __(mo):
    mo.md(rf"# plot")
    return


@app.cell
def __(pd):
    data = pd.read_csv("error_report.csv")
    return data,


@app.cell
def __():
    import matplotlib.pyplot as plt
    return plt,


@app.cell
def __(data):
    data
    return


@app.cell
def __():
    # # import matplotlib.pyplot as plt

    # # 假设 data 是你的 DataFrame

    # # 绘制点线图并加标签
    # plt.plot(data["select_proportion"], data["max_dist1"], 'o-', label=r"$e(D^{d_1 \times d_1})$")
    # plt.plot(data["select_proportion"], data["max_dist2"], 'o-', label=r"$e(D^{d_2 \times d_2})$")
    # plt.plot(data["select_proportion"], data["max_dist3"], 'o-', label=r"$e(D^{d_1 \times d_2})$")

    # # 添加数字标识
    # for i, txt in enumerate(data["max_dist1"][:1]):
    #     plt.annotate("{:.2f}".format(txt), (data["select_proportion"][i], data["max_dist1"][i]))
    # for i, txt in enumerate(data["max_dist2"][:1]):
    #     plt.annotate("{:.2f}".format(txt), (data["select_proportion"][i], data["max_dist2"][i]))
    # for i, txt in enumerate(data["max_dist3"]):
    #     plt.annotate("{:.2f}".format(txt), (data["select_proportion"][i], data["max_dist3"][i]))

    # # 设置X轴标签
    # plt.xlabel(r"Overlap Ratio ($\alpha$)")
    # plt.ylabel("Max Error")

    # # 显示图例
    # plt.legend()
    # plt.title("Using Max normalization (re-scale after normlize)")
    # # 显示图形
    # plt.show()
    return


@app.cell
def __():
    # # import matplotlib.pyplot as plt

    # # 假设 data 是你的 DataFrame

    # # 绘制点线图并加标签
    # plt.plot(data["select_proportion"], data["unnorma_dist1"], 'o-', label=r"$e(D^{d_1 \times d_1})$")
    # plt.plot(data["select_proportion"], data["unnorma_dist2"], 'o-', label=r"$e(D^{d_2 \times d_2})$")
    # plt.plot(data["select_proportion"], data["unnorma_dist3"], 'o-', label=r"$e(D^{d_1 \times d_2})$")

    # # 添加数字标识
    # for _i, _txt in enumerate(data["unnorma_dist1"][:1]):
    #     plt.annotate("{:.2f}".format(_txt), (data["select_proportion"][_i], data["unnorma_dist1"][_i]))
    # for _i, _txt in enumerate(data["unnorma_dist2"][:1]):
    #     plt.annotate("{:.2f}".format(_txt), (data["select_proportion"][_i], data["unnorma_dist2"][_i]))
    # for _i, _txt in enumerate(data["unnorma_dist3"]):
    #     plt.annotate("{:.2f}".format(_txt), (data["select_proportion"][_i], data["unnorma_dist3"][_i]))

    # # 设置X轴标签
    # # plt.xlabel("Overlap Ratio")
    # plt.xlabel(r"Overlap Ratio ($\alpha$)")
    # plt.ylabel("Max Error")

    # # 显示图例
    # plt.legend()
    # plt.title("Do not use Normalization")
    # # 显示图形
    # plt.show()
    return


@app.cell
def __(pd, plt):
    # import pandas as pd
    # import matplotlib.pyplot as plt_

    # 创建数据
    ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    max_12 = [50.7, 71.8, 74.1, 76.9, 88.4, 91.1]  # max_dist1
    max_22 = [97.4]*len(max_12)                    # max_dist2
    max_11 = [98.7]*len(max_12)                    # max_dist3

    data_new = pd.DataFrame({
        "select_proportion": ratios,
        "max_dist1": max_12,
        "max_dist2": max_22,
        "max_dist3": max_11
    })

    # 绘制点线图并加标签
    plt.plot(data_new["select_proportion"], data_new["max_dist1"], 'o-', label=r"$e(D^{d_1 \times d_1})$")
    plt.plot(data_new["select_proportion"], data_new["max_dist2"], 'o-', label=r"$e(D^{d_2 \times d_2})$")
    plt.plot(data_new["select_proportion"], data_new["max_dist3"], 'o-', label=r"$e(D^{d_1 \times d_2})$")

    # 添加数字标识
    for i_, txt_ in enumerate(data_new["max_dist1"]):
        plt.annotate("{:.2f}".format(txt_), (data_new["select_proportion"][i_], data_new["max_dist1"][i_]))
    for i_, txt_ in enumerate(data_new["max_dist2"]):
        plt.annotate("{:.2f}".format(txt_), (data_new["select_proportion"][i_], data_new["max_dist2"][i_]))
    for i_, txt_ in enumerate(data_new["max_dist3"]):
        plt.annotate("{:.2f}".format(txt_), (data_new["select_proportion"][i_], data_new["max_dist3"][i_]))

    # 设置X轴和Y轴标签
    plt.xlabel(r"Overlap Ratio ($\alpha$)")
    plt.ylabel("Average Accuracy")

    # 显示图例
    plt.legend()
    # plt.title("Using Max normalization (re-scale after normalize)")
    plt.title("Link Prediction")

    # 显示图形
    plt.show()

    return data_new, i_, max_11, max_12, max_22, ratios, txt_


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
