
import click
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

fsl = fs = 9

@click.command()
@click.argument('csv-path', type=click.File())
@click.option('--outputpath', '-o', type=click.Path(), default="perf-plot.png")
def perfplot(csv_path, outputpath="perf-plot.png"):
    fig = plt.figure(figsize=(3.3, 3.3))

    data = pd.read_csv(csv_path)
    find = data[data['operation'] == 'find']
    replace = data[data['operation'] == 'replace']
    print(data)

    ax = fig.subplots(ncols=1)
    ax.set_xticks([0, 50000, 100000, 150000, 200000, 250000])
    ax.set_xticklabels(["0", "50K", "100K", "150K", "200K", "250K"])

    ax.set_ylim((-10, 400))
    ax.set_xlim((-15000, 250000))

    ax.grid(which='major', axis='x', linestyle='-', color='0.85', zorder=10)
    ax.grid(which='major', axis='y', linestyle='-', color='0.85', zorder=10)

    ax.set_ylabel("Time (s)")
    ax.set_xlabel("# Atoms")

    ax.scatter(pd.to_numeric(find['num_atoms']), find['process_time_seconds'], label="Find", zorder=50)
    ax.scatter(pd.to_numeric(replace['num_atoms']), replace['process_time_seconds'], label="Find + Replace", zorder=50)

    for row in replace.itertuples():
        print(row)
        ax.annotate(row.repl, (row.num_atoms, row.process_time_seconds), xytext=(0,3), textcoords='offset points', horizontalalignment="center", verticalalignment='bottom', fontsize=7.5, c="black")

    ax.legend()

    fig.savefig(outputpath, dpi=300, bbox_inches='tight')#, transparent=True)
    plt.close(fig)


if __name__ == '__main__':
    perfplot()
