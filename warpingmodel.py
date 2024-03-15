import cv2
import mediapipe as mp
import numpy as np
import trimesh
import matplotlib.pyplot as plt

# Load the 3D model
scene = trimesh.load('11762_Earrings_v1_l2.obj')
 # Invert the y-axis for visualization purposes
angle = np.pi /2# 90 degrees in radians
axis = [-1, 0, 0]   # x-axis

# Apply rotation
for mesh in scene.geometry.values():
    mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, axis))

# Load the image
image_path = 'image.jpg'
image = cv2.imread(image_path)
height, width, _ = image.shape

# Facial mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Convert image to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Facial landmarks
results = face_mesh.process(rgb_image)

# Specify indices of keypoints to draw (e.g., for left and right earlobes)
left_earlobe_index = 234  # Example index (adjust as needed)
right_earlobe_index = 454  # Example index (adjust as needed)
adjustment_y = 0.035
adjustment_x = 0.0050

# Collect keypoints for left and right earlobes
left_earlobe_keypoints = []
right_earlobe_keypoints = []

# Get the keypoints for left and right earlobes
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        if left_earlobe_index < len(face_landmarks.landmark):
            left_earlobe_point = face_landmarks.landmark[left_earlobe_index]
            x_left = int((left_earlobe_point.x - adjustment_x) * width)
            y_left = int((left_earlobe_point.y + adjustment_y) * height)
            left_earlobe_keypoints.append((x_left, y_left))

        if right_earlobe_index < len(face_landmarks.landmark):
            right_earlobe_point = face_landmarks.landmark[right_earlobe_index]
            x_right = int((right_earlobe_point.x + adjustment_x) * width)
            y_right = int((right_earlobe_point.y + adjustment_y) * height)
            right_earlobe_keypoints.append((x_right, y_right))

# Calculate the translation vector based on the midpoint between left and right earlobe keypoints
if left_earlobe_keypoints and right_earlobe_keypoints:
    mid_x = (left_earlobe_keypoints[0][0] + right_earlobe_keypoints[0][0]) // 2
    mid_y = (left_earlobe_keypoints[0][1] + right_earlobe_keypoints[0][1]) // 2
    translation_vector = np.array([mid_x, mid_y, 0])

    # Scale the earring model to fit the distance between earlobes
    model_scale_factor = np.linalg.norm(np.array(left_earlobe_keypoints[0]) - np.array(right_earlobe_keypoints[0]))
    # scaled_model = model.copy()
    mesh.vertices *= model_scale_factor

    # Draw the scaled model onto the image
    for vertex in mesh.vertices.astype(int):
        x, y = vertex[0] + mid_x, vertex[1] + mid_y
        cv2.circle(image, (x, y), 1, (0, 0, 0), -1)

# Display the image with overlaid earring model
cv2.imshow('Face with Earring', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# model.show()