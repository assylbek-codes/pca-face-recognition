from src.data_loading import load_data
from src.face_operations import display_random_face, compute_and_display_mean_face, center_dataset
from src.eigen_operations import compute_matrix_l, compute_eigen_vectors_values_matrix_v
from src.visualization import plot_cumulative_variance, display_grid

def main():
    faces = load_data("data/faces.csv")

    # Display a random face
    display_random_face(faces)

    # Compute and display mean face
    mean_face = compute_and_display_mean_face(faces, display=True)

    # Center dataset
    centered_faces = center_dataset(faces, mean_face)

    # Compute matrix L
    matrix_l = compute_matrix_l(centered_faces)

    # Compute eigenvectors and eigenvalues
    eigenfaces_normalized, eigenvalues = compute_eigen_vectors_values_matrix_v(centered_faces, matrix_l)

    # Plot cumulative variance
    plot_cumulative_variance(eigenvalues)

    # Display grid of first 48 principal components
    display_grid(mean_face, eigenfaces_normalized, grid_size=7, plot_title="6. Plot mean_face and 48 PC in 7x7 grid.png")


if __name__ == "__main__":
    main()
