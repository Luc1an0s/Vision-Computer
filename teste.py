import cv2
import face_recognition


image = cv2.imread("luciano.jpg")
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


landmarks_list = face_recognition.face_landmarks(rgb_image)


for landmarks in landmarks_list:
    for feature, points in landmarks.items():
        for i in range(len(points) - 1):
            pt1 = points[i]
            pt2 = points[i + 1]
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)


cv2.imshow("Linhas do Rosto", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
