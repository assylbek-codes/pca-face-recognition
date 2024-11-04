import numpy as np
import matplotlib.pyplot as plt

def display_face(face_data, title):
    reshaped_face = np.reshape(face_data, (64, 64), order='F')
    plt.figure()
    plt.title(title)
    plt.imshow(reshaped_face, cmap=plt.cm.gray)
    plt.show()

def compute_and_display_mean_face(faces, display=False):
    mean_face = np.mean(faces, axis=0)
    if display:
        display_face(mean_face, title='Mean Face')
    return mean_face

def center_dataset(faces, mean_face):
    centered_faces = faces - mean_face
    return centered_faces

def display_random_face(faces):
    random_index = np.random.choice(faces.shape[0])
    random_face = faces[random_index]
    display_face(random_face, title=f'Random Face {random_index}')
