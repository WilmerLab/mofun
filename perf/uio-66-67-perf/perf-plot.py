
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
    fig = plt.figure(figsize=(6, 6))

    data = pd.read_csv(csv_path)

    ax = fig.subplots(ncols=1)
    # ax.set_xticks(np.arange(len(ranked)))
    # ax.set_xticklabels(ranked["mof"], rotation='vertical', fontsize=fs)

    ax.scatter(data['num-atoms'], data['process-time-seconds'])
    fig.savefig(outputpath, dpi=300, bbox_inches='tight')#, transparent=True)
    plt.close(fig)


if __name__ == '__main__':
    perfplot()
