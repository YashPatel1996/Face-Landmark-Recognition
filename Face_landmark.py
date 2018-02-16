import dlib
import cv2
from imutils import face_utils
import numpy as np
import math
# You can download the required pre-trained face detection model here:
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model = "shape_predictor_68_face_landmarks.dat"
# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)


def distance(x1,x2,y1,y2):
    return math.sqrt(math.pow((x1-x2),2)+math.pow((y1-y2),2))
cap = cv2.VideoCapture(0)
while (True):
    ret, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_faces = face_detector(gray, 1)
    #print("Found {} faces in the image file".format(len(detected_faces)))
    x=0
    y=0
    w=0
    h=0
    #win.set_image(image)
    for i, face_rect in enumerate(detected_faces):
        # Detected faces are returned as an object with the coordinates
        # of the top, left, right and bottom edges
        #print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(),
                                                                                 #face_rect.right(), face_rect.bottom()))

        x = face_rect.left()
        y = face_rect.top()
        w = face_rect.right() - x
        h = face_rect.bottom() - y


        if len(detected_faces)!=0:
            pose_landmarks = face_pose_predictor(image, face_rect)
            pose_landmarks = face_utils.shape_to_np(pose_landmarks)
        if len(detected_faces)!=0:
            eyes_1_pts = pose_landmarks[36:42]
            eyes_2_pts = pose_landmarks[42:48]
            #nose_pts = pose_landmarks[27:36]
            nose_pts_left = np.array([pose_landmarks[27],pose_landmarks[31],pose_landmarks[30]])
            nose_pts_right = np.array([pose_landmarks[27],pose_landmarks[30],pose_landmarks[35]])
            nose_pts_lower = np.array([pose_landmarks[31],pose_landmarks[30],pose_landmarks[35]])
            lips_upper = np.array([pose_landmarks[48],pose_landmarks[49],pose_landmarks[50],pose_landmarks[51],pose_landmarks[52],pose_landmarks[53],pose_landmarks[54],pose_landmarks[60],pose_landmarks[61],pose_landmarks[62],pose_landmarks[63],pose_landmarks[64]])
            lips_lower = np.array([pose_landmarks[54],pose_landmarks[55],pose_landmarks[56],pose_landmarks[57],pose_landmarks[58],pose_landmarks[59],pose_landmarks[48],pose_landmarks[60],pose_landmarks[67],pose_landmarks[66],pose_landmarks[65],pose_landmarks[64]])
            inside_lips = pose_landmarks[60:68]
            eyebrow_left = pose_landmarks[17:22]
            eyebrow_right = pose_landmarks[22:27]
            #mos = np.array([pose_landmarks[48],pose_landmarks[49],pose_landmarks])
            mos = pose_landmarks[48:55]
            jaws_1 = pose_landmarks[1:16]
            jaws_2 = pose_landmarks[2:15]
            jaws_3 = pose_landmarks[3:14]
            jaws_4 = pose_landmarks[4:13]
            jaws_5 = pose_landmarks[5:12]
            jaws_6 = pose_landmarks[6:11]
            eyebrow_left = eyebrow_left.reshape((-1, 1, 2))
            eyebrow_right = eyebrow_right.reshape((-1, 1, 2))
            eyes_1_pts = eyes_1_pts.reshape((-1, 1, 2))
            eyes_2_pts = eyes_2_pts.reshape((-1, 1, 2))
            nose_pts_right = nose_pts_right.reshape((-1, 1, 2))
            nose_pts_lower = nose_pts_lower.reshape((-1, 1, 2))
            nose_pts_left = nose_pts_left.reshape((-1, 1, 2))
            lips_upper = lips_upper.reshape((-1, 1, 2))
            lips_lower = lips_lower.reshape((-1, 1, 2))
            inside_lips = inside_lips.reshape((-1, 1, 2))
            jaws_1 = jaws_1.reshape((-1, 1, 2))
            jaws_2 = jaws_2.reshape((-1, 1, 2))
            jaws_3 = jaws_3.reshape((-1, 1, 2))
            jaws_4 = jaws_4.reshape((-1, 1, 2))
            jaws_5 = jaws_5.reshape((-1, 1, 2))
            jaws_6 = jaws_6.reshape((-1, 1, 2))
            #mos = mos.reshape((-1, 1, 2))
            mos = np.array([pose_landmarks[48],pose_landmarks[31],pose_landmarks[32],pose_landmarks[33],pose_landmarks[34],pose_landmarks[35],pose_landmarks[54],pose_landmarks[53],
                            pose_landmarks[52],pose_landmarks[51],pose_landmarks[50],pose_landmarks[49]])
            cv2.fillPoly(image, [eyes_1_pts],(0,0, 255))
            cv2.fillPoly(image, [eyes_2_pts], (0, 0, 255))
            #cv2.fillPoly(image, [nose_pts_left],(25, 25, 25))
            #cv2.fillPoly(image, [nose_pts_right],(30, 30, 30))
            #cv2.fillPoly(image, [nose_pts_lower],(50, 50, 50))
            cv2.fillPoly(image, [lips_upper],(0, 0, 255))
            cv2.fillPoly(image, [lips_lower],(0, 0, 255))
            cv2.fillPoly(image, [mos],(0,0,0))
            #cv2.fillPoly(image, [inside_lips],(0, 0, 0))
            #cv2.polylines(image,[eyebrow_left],isClosed=False,thickness=4,color=(255,255,255))
            #cv2.polylines(image,[eyebrow_right],isClosed=False,thickness=4,color=(255,255,255))
            cv2.polylines(image,[jaws_1],isClosed=False,thickness=5,color=(0,0,0))
            cv2.polylines(image,[jaws_2],isClosed=False,thickness=9,color=(0,0,0))
            cv2.polylines(image,[jaws_3],isClosed=False,thickness=12,color=(0,0,0))
            cv2.polylines(image,[jaws_4],isClosed=False,thickness=30,color=(0,0,0))
            cv2.polylines(image,[jaws_5],isClosed=False,thickness=45,color=(0,0,0))
            cv2.polylines(image,[jaws_6],isClosed=False,thickness=53,color=(0,0,0))
            #cv2.polylines(image,[mos],isClosed=True,thickness=1,color=(35,255,0))
            #print("this image: ",pose_landmarks[19])
    #image[0:285,0:285]  = img2

            
    cv2.imshow("framw",image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()