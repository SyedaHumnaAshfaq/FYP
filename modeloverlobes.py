import cv2
import trimesh
import numpy as np
import mediapipe as mp

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

# Replace these with your actual keypoints
# x_left, y_left = 100, 150
# x_right, y_left = 200, 150

# Load your earring model (replace 'earring.obj' with your file)
earring_mesh = trimesh.load('11762_Earrings_v1_l2.obj')

# Function to project 3D point to 2D image plane (assuming camera parameters are known)
# def project_3d_to_2d(point_3d):
#   # Replace with your camera calibration matrix and distortion coefficients (if available)
#   camera_matrix = np.eye(3)
#   distortion_coefficients = np.zeros(4)
#   return cv2.projectPoints(point_3d.reshape(1, -1), np.zeros(3), camera_matrix, distortion_coefficients)[0].flatten()


# Function to position earring based on keypoint and desired offset
def position_earring(keypoint_x, keypoint_y, offset_x, offset_y):
  # Get center point of the earring mesh (assuming you know it)
  center_point = earring_mesh.centroid

  # Directly translate the center point based on keypoint and offset
  translation = [keypoint_x + offset_x, keypoint_y + offset_y, 0]  # Assuming Z-offset is minimal

  # Move the entire mesh by the calculated offset
  earring_mesh.translation = translation

  return earring_mesh
# Load your face image (replace 'face.jpg' with your file)
# face_image = cv2.imread('image.jpg')

# Position left earring
left_earring = position_earring(left_earlobe_point.x, left_earlobe_point.y, -5, 0)  # Adjust offset values as needed

# Position right earring (adjust offset for right side)
right_earring = position_earring(right_earlobe_point.x, right_earlobe_point.y, 5, 0)

# Convert the earring mesh to a format suitable for rendering on the image (e.g., OpenCV Point2d list)
def convert_mesh_to_points(mesh):
  points = []
  for vertex in mesh.vertices:
    # Assuming a fixed image plane position for simplicity (adjust Z if needed)
    points.append((vertex[0], vertex[1]))  # Only X and Y coordinates
  return points

# Get points for left and right earrings
left_earring_points = convert_mesh_to_points(left_earring)
right_earring_points = convert_mesh_to_points(right_earring)

# Draw the earring points on the face image
cv2.polylines(image, [np.array(left_earring_points)], True, (0, 0, 255), 2)  # Blue for left
cv2.polylines(image, [np.array(right_earring_points)], True, (255, 0, 0), 2)  # Red for right

# Display the final image with positioned earrings
cv2.imshow('Earring on Face', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
