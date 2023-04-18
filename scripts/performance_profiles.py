from matplotlib.ticker import AutoMinorLocator
import pandas as pd
from collections import *
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import copy

# import scales

import scipy.stats
import math
import numpy as np

#import tikzplotlib

max_objective = 2147483647
no_algo_solved_ratio = max_objective-1
imbalanced_ratio = max_objective-2
timeout_ratio = max_objective-3

performance_profile_fraction_scaling = 1

plt.rc('text', usetex=True)
plt.rc('font', family='libertine')
plt.rc('font', size=12)
sb.set_context("paper")
plt.rcParams['text.latex.preamble'] = R'\usepackage{pifont}'

# Returns for each algorithm the tau's at which the number of instances solved within tau*best jumps
def performance_profiles(algos, instances, input_df, objective="judiciousLoad"):
    df = input_df[input_df.algorithm.isin(algos)].copy()

    instance_grouper = ["instance"]
    index = df.groupby(instance_grouper + ["algorithm"]).mean()
    best_per_instance = index[objective].groupby(level=instance_grouper).min()

    ratios = defaultdict(list)
    n = len(instances)
    unsolved = set()
    solved = defaultdict(list)

    for instance in instances:

        if not instance in best_per_instance:
            unsolved.add(instance)
            # print("no algo solved", instance)
            # for algo in algos:
            # 	ratios[algo].append(no_algo_solved_ratio)
            continue

        best = best_per_instance.loc[instance]

        for algo in algos:
            key = instance,algo
            solved[algo].append(instance)
            obj = index.loc[key][objective]
            if best != 0:
                r = obj / best
                if r < 1:
                    print("r < 1", instance, algo, obj, best, r)
            else:
                if obj == 0:
                    r = 1
                else:
                    r = obj + 1
            ratios[algo].append(r)
            #if r > 1.1:
            #	print(algo, G, k, r)

    max_ratio = max( max(ratios[algo]) for algo in algos )
    print("max ratio = ", max_ratio)
    for algo in algos:
        print(algo, "solved", len(solved[algo]), "instances. gmean performance ratio", scipy.stats.gmean(ratios[algo]))

    last_drawn_ratio = min(imbalanced_ratio, max_ratio)

    output = []
    for algo in algos:
        ratios[algo].sort()
        num_best = sum(map(lambda ratio : 1 if ratio == 1 else 0, ratios[algo]))
        print(algo, num_best)
        for i, (r_prev, r) in enumerate(zip(ratios[algo], ratios[algo][1:])):
            if r_prev != r:
                output.append((algo, i * performance_profile_fraction_scaling / n, r_prev))		# last occurence of r_prev

        # output.append((algo, (len(ratios[algo])-1) * performance_profile_fraction_scaling / n, ratios[algo][-1]))
        if all(x == 1.0 for x in ratios[algo]):
            output.append((algo, 1.0, 1.0))

        # output.append((algo, len(ratios[algo]) * performance_profile_fraction_scaling / n, last_drawn_ratio))	# draw the step function to the rightmost x-value


    output_df = pd.DataFrame(data=output, columns=["algorithm", "fraction", "ratio"])
    return output_df


def plot(plotname, df, colors, display_legend="Yes", title=None,
        grid=True, width_scale=1.0, figsize=None, legend_fontsize=10):
    algos = df.algorithm.unique()

    max_ratio = df.ratio.max()
    max_actual_ratio = df[df.ratio < timeout_ratio].ratio.max()
    show_timeout_tick = max_ratio >= timeout_ratio
    show_imbalanced_tick = max_ratio >= imbalanced_ratio

    base = math.ceil(math.log10(max_actual_ratio))
    if (math.log10(max_actual_ratio).is_integer()):
        base += 1
    remapped_timeout_ratio = 10 ** base
    remapped_imbalanced_ratio = 10 ** (base + 1)
    if show_timeout_tick:
        print("do remap. new timeout ratio", remapped_timeout_ratio)
        df.ratio.replace(to_replace={timeout_ratio : remapped_timeout_ratio, imbalanced_ratio : remapped_imbalanced_ratio}, inplace=True)


    last_drawn_ratio = df.ratio.max()
    bb = [0.995, 1.1, 2, last_drawn_ratio * 1.05]
    ymax = 1.01 * performance_profile_fraction_scaling

    nbuckets = len(bb) - 1
    for i in range(len(bb) - 1):
        if bb[i] > max_ratio:
            nbuckets = i
            break
    print("nbuckets=", nbuckets, "max_ratio=", max_ratio)
    if nbuckets < 1:
        print("Warning nbuckets = 0. Aborting")
        return

    fig = plt.figure(figsize=figsize)
    gs = grd.GridSpec(nrows=1, ncols=nbuckets, wspace=0.0, hspace=0.0, width_ratios=[1.0/ nbuckets for i in range(nbuckets)])
    axes = [plt.subplot(gs[i]) for i in range(nbuckets)]



    for algo in algos:
        algo_df = df[df.algorithm == algo]
        for i, ax in enumerate(axes):
            ax.plot(algo_df["ratio"], algo_df["fraction"], color=colors[algo], lw=2.2, label=algo)

    if display_legend == "Yes":
        axes[-1].legend(fancybox=True, framealpha=1.0, fontsize=legend_fontsize)
        #if len(algos) < 5:
        #	plt.legend(ncol=1, fancybox=True, framealpha=1.0, loc='lower right')
        #else:
        #	if width_scale > 1.0:
        #		ncols = 5
        #	else:
        #		ncols = 2
        #	axes[0].legend(ncol=ncols, fancybox=False, frameon=False, loc='upper left', bbox_to_anchor=(-0.12,-0.15))
    elif display_legend == "Externalize":
        fig_leg = plt.figure()
        fig_leg.legend(*axes[0].get_legend_handles_labels(), loc='center', ncol=2, fancybox=True, framealpha=1.0)
        fig_leg.savefig(display_legend + ".pdf", bbox_inches="tight")
        plt.close(fig_leg)
    elif display_legend == "wide":
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc=(0.3, 0.3), fancybox=True, framealpha=1.0)
    elif display_legend == "lower right":
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc=(0.55, 0.2), fancybox=True, framealpha=0.8)
        # axes[1].legend(fancybox=True, framealpha=1.0, fontsize=legend_fontsize)
    elif display_legend == "below":
        ncols = 2
        axes[0].legend(ncol=ncols, fancybox=False, frameon=False, loc='upper left', bbox_to_anchor=(-0.12,-0.19))

    for i in range(nbuckets):
        axes[i].set_xlim(bb[i], bb[i+1])
        axes[i].set_ylim(-0.01, ymax)
        if grid:
            axes[i].grid(visible=True, axis='both', which='major', ls='dashed')

    if nbuckets > 1:
        if width_scale > 1.0:
            x0 = [1, 1.025, 1.05, 1.075, 1.1]
            axes[0].set_xticks(x0)
            axes[0].set_xticklabels(x0)
            x1 = [1.2, 1.4, 1.6, 1.8]
            axes[1].set_xticks(x1)
            axes[1].set_xticklabels(x1)
        else:
            axes[0].set_xticks([1, 1.05, 1.1])
            axes[0].xaxis.set_minor_locator(AutoMinorLocator(4))
            axes[0].tick_params(which='minor', length=4)
            ax1_xticks = [1.5, 2.0]
            if nbuckets == 3 and 2 in axes[2].get_xticks():
                ax1_xticks.remove(2.0)
            axes[1].set_xticks(ax1_xticks)
            axes[1].set_xticklabels(ax1_xticks)


    if last_drawn_ratio >= 500 or show_timeout_tick:
        axes[nbuckets - 1].set_xscale('log')
        #axes[nbuckets - 1].set_xscale('fifthroot')
    if show_timeout_tick:
        ticks = [10 ** i for i in range(1, base)]
        tick_labels = copy.copy(ticks)
        ticks.append(remapped_timeout_ratio)
        tick_labels.append(R'\ding{99}')
        if show_imbalanced_tick:
            ticks.append(remapped_imbalanced_ratio)
            tick_labels.append(R'\ding{56}')
        axes[nbuckets - 1].set_xticks(ticks)
        axes[nbuckets - 1].set_xticklabels(tick_labels)

    if nbuckets == 3 and 2.5 in axes[2].get_xticks():
        axes[2].set_xticks(np.arange(4.0, last_drawn_ratio, step=2.0))


    axes[0].set_ylabel('fraction of instances')
    for ax in axes:			# generate the ticks for each bucket, so that the grid looks good!
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    for ax in axes[1:]:
        ax.set_yticklabels(ax.get_yticklabels(), visible=False)
        ax.yaxis.set_ticks_position('none')
    if nbuckets == 2:
        axes[0].set_xlabel('performance ratio', x=1.05)
    elif nbuckets == 3:
        axes[1].set_xlabel('performance ratio')
    else:
        axes[0].set_xlabel('performance ratio')

    if title != None:
        axes[nbuckets-1].text(0.5, 0.2, title, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=dict(facecolor='white'))

    #fig.tight_layout()
    fig.savefig(plotname + "_performance_profile.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


if __name__ == '__main__':
    import sys, commons
    plot_name = sys.argv[1]
    load_field = sys.argv[2]
    files = sys.argv[3:]
    df = commons.read_files(files)
    df.drop(df[df.algorithm.str.contains("MT-KaHyPar-Graph-Q-F 2")].index, inplace=True)
    df.drop(df[df.algorithm.str.contains("MT-KaHyPar-Graph-Q-F 4")].index, inplace=True)
    df.drop(df[df.algorithm.str.contains("MT-KaHyPar-Graph-Q-F 8")].index, inplace=True)
    df.drop(df[df.algorithm.str.contains("MT-KaHyPar-Graph-Q-F 16")].index, inplace=True)
    df.drop(df[df.algorithm.str.contains("MT-KaHyPar-Q-F 2")].index, inplace=True)
    df.drop(df[df.algorithm.str.contains("MT-KaHyPar-Q-F 4")].index, inplace=True)
    df.drop(df[df.algorithm.str.contains("MT-KaHyPar-Q-F 8")].index, inplace=True)
    df.drop(df[df.algorithm.str.contains("MT-KaHyPar-Q-F 16")].index, inplace=True)
    df.drop(df[df.algorithm.str.contains("MT-KaHyPar-Graph-D 2")].index, inplace=True)
    df.drop(df[df.algorithm.str.contains("MT-KaHyPar-Graph-D 4")].index, inplace=True)
    df.drop(df[df.algorithm.str.contains("MT-KaHyPar-Graph-D 8")].index, inplace=True)
    df.drop(df[df.algorithm.str.contains("MT-KaHyPar-Graph-D 16")].index, inplace=True)
    df.drop(df[df.algorithm.str.contains("MT-KaHyPar-D 2")].index, inplace=True)
    df.drop(df[df.algorithm.str.contains("MT-KaHyPar-D 4")].index, inplace=True)
    df.drop(df[df.algorithm.str.contains("MT-KaHyPar-D 8")].index, inplace=True)
    df.drop(df[df.algorithm.str.contains("MT-KaHyPar-D 16")].index, inplace=True)

    algos = commons.infer_algorithms_from_dataframe(df)
    instances = df.instance.unique()
    colors = commons.construct_new_color_mapping(algos)
    final_portfolio_name = set(algos).intersection(set(["final_portfolio", "final_portfolio_opt"])).pop()
    for algo in algos:
        if algo == final_portfolio_name: continue
        plot_name_prefix = f"{plot_name}_{final_portfolio_name}_{algo}_{load_field}"
        ratios_df = performance_profiles([final_portfolio_name, algo], instances, df, objective=load_field)
        plot(plot_name_prefix, ratios_df, colors=colors)

    plot_name = f"{plot_name}_{final_portfolio_name}_{load_field}"
    ratios_df = performance_profiles(algos, instances, df, objective=load_field)
    plot(plot_name, ratios_df, colors=colors)
