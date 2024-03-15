import sys
import cv2
import mediapipe as mp
import numpy as np



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


# Display the image with keypoints
cv2.imshow('Face with Keypoints', image)
cv2.waitKey(0)
cv2.destroyAllWindows()








