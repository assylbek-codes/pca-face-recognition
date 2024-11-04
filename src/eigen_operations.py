import numpy as np

def compute_matrix_l(centered_faces):
    centered_faces_transposed = centered_faces.T
    matrix_l = np.matmul(centered_faces, centered_faces_transposed)
    return matrix_l

def compute_eigen_vectors_values_matrix_v(centered_faces, matrix_l):
    eigenvalues, eigenvectors_L = np.linalg.eig(matrix_l)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors_L = eigenvectors_L[:, idx]

    centered_faces_transposed = centered_faces.T
    eigenvectors_V = centered_faces_transposed @ eigenvectors_L

    norms = np.linalg.norm(eigenvectors_V, axis=0)
    eigenvectors_V_normalized = eigenvectors_V / norms

    eigenvectors_V_normalized = eigenvectors_V_normalized.T
    return eigenvectors_V_normalized, eigenvalues
