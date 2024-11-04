import numpy as np
import matplotlib.pyplot as plt

def display_grid(first_face, faces_in_grid, grid_size, plot_title):
    _, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    reshaped_first_face = np.reshape(first_face, (64, 64), order='F')
    axes[0, 0].imshow(reshaped_first_face, cmap='gray')
    axes[0, 0].axis('off')

    for i in range(grid_size*grid_size-1):
        reshaped_face = np.reshape(faces_in_grid[i], (64, 64), order='F')
        row, col = (i + 1) // grid_size, (i + 1) % grid_size
        axes[row, col].imshow(reshaped_face, cmap='gray')
        axes[row, col].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(plot_title)
    plt.show()
    plt.close()

def plot_cumulative_variance(eigenvalues):
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='b')
    plt.title('Cumulative Proportion of Variance Explained by Principal Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Proportion of Variance Explained')
    plt.grid(True)
    plt.savefig("9. Cumulative Proportion of Variance Explained by Principal Components.png")
    plt.show()
    plt.close()
