import os
import cv2

DATA_DIR = './modelCreation/data'

labels_dict = {0: 'A', 1: 'B', 2: 'C',3:'D',4:'E',
               5: 'F', 6: 'G', 7: 'H',8:'I',9:'J',
               10: 'K', 11: 'L', 12: 'M',13:'N',14:'O',
               15: 'P',16: 'Q', 17: 'R', 18: 'S',19:'T',
               20:'U',21: 'V', 22: 'W', 23: 'X',24:'Y',25:'Z'} 


classes_number = 0
dataset_size = 200

cap = cv2.VideoCapture(0)

if not os.path.exists(os.path.join(DATA_DIR, str(classes_number))):
        os.makedirs(os.path.join(DATA_DIR, str(classes_number)))

counter = 0
while counter < dataset_size:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.waitKey(25)
    cv2.imwrite(os.path.join(DATA_DIR, str(classes_number), '{}.jpg'.format(counter)), frame)

    counter += 1

cap.release()
cv2.destroyAllWindows()
