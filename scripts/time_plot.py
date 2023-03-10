import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import scipy.stats.mstats
import numpy as np

def plot(df, colors, algo_order, plot_name='', field='totalPartitionTime'):
    num_k = len(df.k.unique())
    fig, ax = plt.subplots(num_k, figsize=(7, num_k * 5))
    boxprops = dict(fill=False,edgecolor='black', linewidth=0.9, zorder=2)
    rem_props = dict(linestyle='-', linewidth=0.9, color='black')

    for i, k in enumerate(sorted(df.k.unique())):
        plot_df = df[df.k == k]
        # print(plot_df)
        strip_plot = sb.stripplot(y=field, x="algorithm", data=plot_df,
                jitter=0.3, size=1.5, edgecolor="gray", alpha=0.4,
                ax=ax[i], zorder=1,
                palette=colors, order=algo_order
                )
        box_plot = sb.boxplot(y=field, x="algorithm", data=plot_df,
                width=0.5, showfliers=False,
                palette=colors, order=algo_order,
                boxprops=boxprops, whiskerprops=rem_props, medianprops=rem_props, meanprops=rem_props, flierprops=rem_props, capprops=rem_props,
                ax=ax[i], zorder=2)

        plt.setp(ax[i].xaxis.get_majorticklabels(), rotation=20, ha="right", rotation_mode="anchor")
        #ax.set_yscale('cuberoot')
        #ax.set_yscale('fifthroot')
        ax[i].set_yscale('log')
        ax[i].set_title(f"k={k}", x=0.9, y=0.9)
        #ax.set(yticks=[0, 1, 20, 75, 250, 750, 2500, 10000, 28800])
        ax[i].set_ylabel('Time [s]')
        ax[i].xaxis.label.set_visible(False)

        gmean_string = "\n".join(["Gmean time"] + [algo + ": " + str(scipy.stats.gmean(plot_df[plot_df.algorithm == algo][field])) for algo in algos])
        ax[i].text(1.1, 0, gmean_string, transform = ax[i].transAxes)

    if plot_name == '':
        file_name = 'runtime_plot.pdf'
    else:
        file_name = plot_name + '_runtime_plot.pdf'

    fig.savefig(file_name, bbox_inches="tight", pad_inches=0.0)


def aggregate_dataframe_by_arithmetic_mean_per_instance(df, field='totalPartitionTime'):
    return df.groupby(["graph", "k", "epsilon", "algorithm"]).mean()[field].reset_index()


def print_gmean_times(df, field='totalPartitionTime'):
    algos = df.algorithm.unique()
    for algo in algos:
        print("Algo", algo, "gmean time", scipy.stats.mstats.gmean( df[df.algorithm==algo][field] ))
        print("Algo", algo, "max time", np.max( df[df.algorithm==algo][field] ))

if __name__ == '__main__':
    import sys, commons
    plot_name = sys.argv[1]
    field = sys.argv[2]
    files = sys.argv[3:]

    # df = pd.concat(map(pd.read_csv, files))
    df = commons.read_files(files)
    # commons.conversion(df)
    # df = df[df.failed == "no"]

    averaged_runtimes = aggregate_dataframe_by_arithmetic_mean_per_instance(df, field)

    algos = commons.infer_algorithms_from_dataframe(df)
    plot(averaged_runtimes, colors=commons.construct_new_color_mapping(algos), algo_order=algos, plot_name=plot_name, field=field)
    print_gmean_times(averaged_runtimes, field)
