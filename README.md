# Face Recognition using PCA

## Project Overview
This project focuses on implementing Principal Component Analysis (PCA) for face recognition. The main objective is to analyze and visualize facial data, compute eigenfaces, and reconstruct faces using a reduced number of principal components. This project uses linear algebra and statistical analysis techniques to represent high-dimensional data in a lower-dimensional space for computational efficiency. It also provides visualizations to help understand how PCA works for facial recognition.

## Getting Started
### Prerequisites
- Python 3.x
- Required Python libraries listed in `requirements.txt`

To install the dependencies, run:
```bash
pip install -r requirements.txt
```

### Directory Structure
The project directory is organized as follows:
```
face-recognition-pca/
├── data/
│   └── faces.csv
├── src/
│   ├── data_loading.py
│   ├── face_operations.py
│   ├── eigen_operations.py
│   ├── visualization.py
│   └── main.py
├── plots/
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```
- `data/`: Contains the input data file (`faces.csv`) for the project.
- `src/`: Contains source code modules for data loading, face operations, eigen computation, and visualization.
- `plots/`: Stores any generated plots.
- `README.md`: This file, explaining the project.
- `requirements.txt`: Python dependencies.
- `.gitignore`: File to ignore unnecessary files in version control.
- `LICENSE`: Project license.

### Running the Project
To run the project, use the following command:
```bash
python src/main.py
```
This script will load the facial dataset, compute the mean face, apply PCA to find the principal components, and visualize the results, including cumulative variance and reconstructed faces.

## How the Code Works
### 1. Data Loading
The `data_loading.py` file contains the `load_data(filename)` function, which reads the facial data from a CSV file and reshapes it for further processing. The data consists of 400 images, each represented as a flattened vector of 4096 pixel values.

### 2. Face Operations
The `face_operations.py` file contains several functions:
- **`display_face(face_data, title)`**: Displays an individual face.
- **`compute_and_display_mean_face(faces, display)`**: Computes and optionally displays the mean face of the dataset.
- **`center_dataset(faces, mean_face)`**: Centers the dataset by subtracting the mean face from each individual face.
- **`display_random_face(faces)`**: Displays a randomly selected face from the dataset.

### 3. Eigen Operations
The `eigen_operations.py` file provides functions to compute eigenvectors and eigenvalues from the centered dataset:
- **`compute_matrix_l(centered_faces)`**: Computes matrix L from the centered dataset.
- **`compute_eigen_vectors_values_matrix_v(centered_faces, matrix_l)`**: Computes the eigenvalues and normalized eigenvectors (eigenfaces).

### 4. Visualization
The `visualization.py` file includes functions to plot and visualize the results:
- **`display_grid(first_face, faces_in_grid, grid_size, plot_title)`**: Displays a grid of faces, including the mean face and the first principal components.
- **`plot_cumulative_variance(eigenvalues)`**: Plots the cumulative variance explained by the principal components.

### 5. Main Script
The `main.py` script orchestrates the entire flow, from loading the data to visualizing the results, and serves as the entry point for running the project.

## Example Usage
This project reads the `faces.csv` file, which contains 400 facial images, each flattened into 4096 pixels. The PCA technique is then applied to reduce dimensionality and extract the most significant features of the dataset. Users can visualize the mean face, the principal components, and face reconstructions using a selected number of principal components.

## Features
- Load facial image data from a CSV file.
- Visualize individual and mean faces.
- Apply PCA to compute eigenfaces.
- Reconstruct faces using a reduced number of principal components.
- Visualize the cumulative variance explained by the principal components.

## Contributing
Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request. For major changes, please open an issue to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any questions, feel free to reach out via GitHub.

