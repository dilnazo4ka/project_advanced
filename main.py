import tensorflow
from imutils import face_utils
from imutils.face_utils import FaceAligner
import imutils
import time
import cv2
import dlib
import os
import numpy as np

from tensorflow.python.keras.models import model_from_json
from tensorflow.python.keras.models import load_model
json_file = open('../model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model (2).h5")
print("Loaded model from disk")
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=64)
people = ['jake', 'sunghoon', 'jungwon', 'jay', 'sunoo', 'heeseung',  'niki']
text_list = ['cd96684c-5f72-4362-a7b2-85279a67d4e1.jpeg', 'zxby0MNVpso.jpeg', '-1579754537.jpg','EN_DIMENSION_014.jpeg'
             , 'photo1676812381.jpeg', '857c0a3b-7af5-4833-afbc-394783622ad6.jpeg', 'photo1653649807.jpeg', '05ZBO_5f.jpg']
for i in text_list:
    while True:
        key = cv2.waitKey(1) & 0xFF
        # e for continue
        if key == ord("e"):
            break
        img_data = cv2.imread(i)
        gray_frame = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        rects = detector(gray_frame, 0)

        for rect in rects:
            faceAligned = fa.align(img_data, gray_frame, rect)
            faceAligned = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
            faceAligned = np.array(faceAligned)
            faceAligned = faceAligned.astype('float32')
            faceAligned /= 255.0
            faceAligned = np.expand_dims([faceAligned], axis=3)

            Y_pred = loaded_model.predict(faceAligned)
            for index, value in enumerate(Y_pred[0]):
                result = people[index] + ': ' + str(int(value * 100)) + '%'
                cv2.putText(img_data, result, (14, 15 * (index + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 0), 1)
            # draw rect around face
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(img_data, (x, y), (x + w, y + h), (255, 0, 255), 2)
            # draw person name
            result = np.argmax(Y_pred, axis=1)
            cv2.putText(img_data, people[result[0]], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)
        # show the frame
        cv2.imshow("Frame", img_data)

# do a bit of cleanup
cv2.destroyAllWindows()