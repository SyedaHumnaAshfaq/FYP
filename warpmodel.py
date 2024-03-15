import cv2
import mediapipe as mp
import numpy as np
import sys
sys.path.append('G:\FYP\FYP_1.0\myenv\Lib\site-packages')  # Replace with actual path
import pyassimp
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

def load_earring_model_parts():
    parts = []
    for part_file in ['part1.fbx', 'part2.fbx', 'part3.fbx']:
        scene = pyassimp.load(part_file)
        parts.extend(scene.meshes)
    return parts

def calculate_transformation_matrices(earlobe_keypoints, earlobe_indices):
    transformation_matrices = []

    for i, earlobe_index in enumerate(earlobe_indices):
        # Extract the coordinates of the earlobe keypoint
        earlobe_x, earlobe_y = earlobe_keypoints[i]

        # Define target position and orientation for the earring model part
        # This could involve simple translations, rotations, and scaling as needed
        # For simplicity, let's assume a simple translation for now
        target_position = (earlobe_x, earlobe_y, 0)  # Coordinates in 3D space

        # Calculate the translation matrix
        translation_matrix = np.array([
            [1, 0, 0, target_position[0]],
            [0, 1, 0, target_position[1]],
            [0, 0, 1, target_position[2]],
            [0, 0, 0, 1]
        ])

        # Add the translation matrix to the list of transformation matrices
        transformation_matrices.append(translation_matrix)

    return transformation_matrices

# Example usage:
# transformation_matrices = calculate_transformation_matrices(earlobe_keypoints, [0, 1])


def warp_earring_model(face_image, earring_model_parts, transformation_matrices):
    # Initialize OpenGL window and viewport
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutCreateWindow(b"Earring Model on Face")
    glViewport(0, 0, face_image.shape[1], face_image.shape[0])

    # Initialize OpenGL matrices for projection and modelview
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (face_image.shape[1] / face_image.shape[0]), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)

    # Enable depth testing for correct rendering order
    glEnable(GL_DEPTH_TEST)

    # Loop over each earring model part and render it
    for i, earring_model_part in enumerate(earring_model_parts):
        # Apply the transformation matrix to position the model part
        glPushMatrix()
        glMultMatrixf(transformation_matrices[i])

        # Render the earring model part
        glBegin(GL_TRIANGLES)
        for face in earring_model_part.faces:
            for vertex_index in face:
                vertex = earring_model_part.vertices[vertex_index]
                glColor3f(1.0, 0.0, 0.0)  # Example color (red)
                glVertex3fv(vertex)
        glEnd()

        # Restore the transformation matrix
        glPopMatrix()

    # Swap buffers to display the rendered image
    glutSwapBuffers()

    # Start the main OpenGL loop
    glutMainLoop()

#face mesh
mp_face_mesh=mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh()

#image
image=cv2.imread('image.jpg')
height,width, _=    image.shape
print("height , width",height,width)
rgb_image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

#Facial landmarks

results=face_mesh.process(rgb_image)

# Specify indices of keypoints to draw (e.g., for left and right earlobes)
left_earlobe_index = 234  # Example index (adjust as needed)
right_earlobe_index = 454  # Example index (adjust as needed)
adjustment_y = 0.035
adjustment_x = 0.0050
# Draw keypoints on the image
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Draw left earlobe keypoint
        if left_earlobe_index < len(face_landmarks.landmark):
            left_earlobe_point = face_landmarks.landmark[left_earlobe_index]
            x_left = int((left_earlobe_point.x-adjustment_x) * width)
            y_left = int((left_earlobe_point.y+adjustment_y) * height)
            cv2.circle(image, (x_left, y_left), 3, (0, 255, 0), -1)

        # Draw right earlobe keypoint
        if right_earlobe_index < len(face_landmarks.landmark):
            right_earlobe_point = face_landmarks.landmark[right_earlobe_index]
            x_right = int((right_earlobe_point.x+adjustment_x) * width)
            y_right = int((right_earlobe_point.y+adjustment_y) * height)
            cv2.circle(image, (x_right, y_right), 3, (0, 255, 0), -1)

earlobe_keypoints = [(x_left, y_left), (x_right, y_right)]
earring_model_parts = load_earring_model_parts()
transformation_matrices = calculate_transformation_matrices(earlobe_keypoints, [0, 1])

# Warp earring model onto face image
warped_image = warp_earring_model(image.copy(), earring_model_parts, transformation_matrices)


# Display the image with keypoints
cv2.imshow('Face with Keypoints', image)
cv2.imshow('Warped Earring Model', warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()








