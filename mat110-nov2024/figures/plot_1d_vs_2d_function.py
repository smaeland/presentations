import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np

DEBUG = False
#DEBUG = True

def plot_3d():
    
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(projection='3d')
    X, Y, Z = axes3d.get_test_data(0.01)

    # Plot the 3D surface
    ax.plot_surface(X, Y, Z, cmap='inferno')

    ax.set(xlim=(-30, 30), ylim=(-30, 30), zlim=(-100, 100))
    ax.set_xlabel('X', fontsize=30)
    ax.set_ylabel('Y', fontsize=30)
    ax.set_zlabel('Z', fontsize=30)

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.tight_layout()

    if DEBUG:
        plt.show()
    else:
        fig.savefig('3d_plot_example.png')


def plot_2d():

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot()
    Y = np.arange(-3.0, 3.0, 0.01)

    Z1 = np.exp(-(Y**2) / 2) / (2 * np.pi)
    Z2 = (np.exp(-(((- 1) / 1.5)**2 + ((Y - 1) / 0.5)**2) / 2) /
          (2 * np.pi * 0.5 * 1.5))
    Z = Z2 - Z1

    plt.plot(Y, Z, color='#8E24AA', linewidth=5)
    ax.set_xlabel('X', fontsize=30)
    ax.set_ylabel('Y', fontsize=30, rotation=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tight_layout()

    if DEBUG:
        plt.show()
    else:
        fig.savefig('2d_plot_example.png')


if __name__ == '__main__':

    plot_2d()
    plot_3d()

