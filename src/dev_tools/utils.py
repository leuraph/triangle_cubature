import numpy as np


def generate_random_triangle():
    """
    returns a random 2d triangle's vertices

    coords: np.ndarray
        a 3x2 array containing the triangles vertices
        in counter-clockwise order
    """
    # Step 1: Get three random angles strictly between 0 and 2*pi
    angles = np.sort(np.random.uniform(0, 2*np.pi, 3))

    # Step 2: Generate three random lengths strictly between 0 and 1
    lengths = np.random.uniform(0, 1, 3)

    # Step 3: Generate three coordinates with these angles and lengths
    x_coords = lengths * np.cos(angles)
    y_coords = lengths * np.sin(angles)
    coords = np.vstack((x_coords, y_coords)).T

    # Step 4: Generate two random stretches in x and y direction
    stretch_x = np.random.uniform(0.5, 2.0)  # Random stretch factor for x
    stretch_y = np.random.uniform(0.5, 2.0)  # Random stretch factor for y

    # Stretch the coordinates
    coords[:, 0] *= stretch_x
    coords[:, 1] *= stretch_y

    # Step 5: Generate two random displacements in x and y direction
    displacement_x = np.random.uniform(-1, 1)  # Random displacement for x
    displacement_y = np.random.uniform(-1, 1)  # Random displacement for y

    # Displace the coordinates
    coords[:, 0] += displacement_x
    coords[:, 1] += displacement_y

    return coords
