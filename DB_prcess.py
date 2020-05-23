from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

model = load_model(os.path.join(os.getcwd(), 'Ashrafi.h5'))

def dis_name(number):
    name_of_dis = ''
    if number == 0:
        name_of_dis = 'Dog'
    elif number == 1:
        name_of_dis = 'Cat'
    else:
        name_of_dis = 'Bird'

    return name_of_dis


def dis_classfi(img):
    eye = img[0]
    fit_img = cv2.resize(eye, (70,70))
    process_img = np.array(fit_img).reshape(-1,70,70,3)
    value = model.predict_classes(process_img)
    got_name = dis_name(value)
    return got_name
