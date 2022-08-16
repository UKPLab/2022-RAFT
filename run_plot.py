import sys
import pickle
from rational.torch import Rational
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

sns.set_style("whitegrid")
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# plt.rcParams['font.size'] = 20

# colors = ["#0D63AE", "#B4C6E0", "#ED1C24", "#F8B2AE", "#AA5A2A"]



def plot_layers_per_task(file_folder, save_folder, selection= [0,1,5,10,11], subplots=None, plot_size=None):
    for file in os.listdir(file_folder):
        filepath = os.path.join(file_folder, file)

        Rational.list = pickle.load(open(filepath, "rb"))

        task = filepath.split("/")[-1].split("_")[2]

        Rational.distribution_display_mode = "points"

        if not plot_size:
            ax = plt.axes()
        else:
            fig, axes = plt.subplots(nrows=plot_size[0], ncols=plot_size[1], constrained_layout=True)
        # fig.tight_layout() 
        for rat_n in selection:
            rat = Rational.list[rat_n]
            if plot_size:
                print(rat_n//plot_size[1], rat_n%plot_size[1])
                ax = axes[rat_n//plot_size[1], rat_n%plot_size[1]]
            # ax.set_ylim(-0.3, 1.5)
            if rat_n != 12:
                # print('rat_n', rat_n)
                rat.func_name = f"layer {rat_n+1}"
            else:
                rat.func_name = f"pooler_layer"
            rat.show(x=np.arange(-4, 4, 0.01), axis=ax, display=False)
            if not plot_size:
                ax.legend()
            ax.set_title(f"layer {rat_n+1}", fontsize=10)
            save_path = os.path.join(save_folder, task+'.svg')
            plt.savefig(save_path, bbox_inches='tight')
        # plt.title("")
            # else:
                # plt.show()

# tasks per layer
def tasks_per_layer(file_folder,save_folder):
    for i in range(12):
        fig, ax = plt.subplots(1, 1)
        for file in os.listdir(file_folder):
            filepath = os.path.join(file_folder, file)
            task = file.split("_")[2]
            print(task)

            Rational.list = pickle.load(open(filepath, "rb"))
            # Rational.show_all(x=torch.arange(-2, 2, 0.01), axes=axes, display=False, colors=[color]*12)
            rat = Rational.list[i]
            rat.func_name = task
            rat.show(x=np.arange(-4, 4, 0.01), axis=ax, display=False)
            ax.legend()
            ax.set_title(f"Rational functions in layer {i+1} across different tasks")
            save_path = os.path.join(save_folder, f'layer_{i}.svg')
            plt.savefig(save_path, bbox_inches='tight')

        # ax.set_ylim(-4, 4)
# ax = plt.gca()
# plt.suptitle("Rational activation functions", y=1.03)

if __name__ == '__main__':
    file_folder = sys.argv[1]
    save_folder = sys.argv[2]
    # tasks_per_layer(file_folder,save_folder)
    plot_layers_per_task(file_folder, save_folder, selection=list(range(12)), plot_size=(3,4))

