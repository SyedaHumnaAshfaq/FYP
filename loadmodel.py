import trimesh
import numpy as np
import matplotlib.pyplot as plt

# Load the 3D model
model = trimesh.load('11762_Earrings_v1_l2.obj')
 # Invert the y-axis for visualization purposes
angle = np.pi /2# 90 degrees in radians
axis = [-1, 0, 0]   # x-axis

# Apply rotation
model.apply_transform(trimesh.transformations.rotation_matrix(angle, axis))
# Visualize the 3D model
model.show()
