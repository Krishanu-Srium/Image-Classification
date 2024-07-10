import joblib
import json
import numpy as np
import base64
import cv2
from sklearn.svm import SVC
import pywt

__class_name_to_number = {}
__class_number_to_name = {}
__model = None

def w2d(img, mode='db1', level=1):
    imArray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H

def classify_image(image_base64_data, file_path=None):
    print("Classify image called")
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)
    if not imgs:
        print("No faces detected")
        return []

    result = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
        len_image_array = 32 * 32 * 3 + 32 * 32
        final = combined_img.reshape(1, len_image_array).astype(float)
        prediction = __model.predict(final)[0]
        probability = np.around(__model.predict_proba(final) * 100, 2).tolist()[0]
        
        result.append({
            'class': class_number_to_name(prediction),
            'class_probability': probability,
            'class_dictionary': __class_name_to_number
        })
    return result

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    print("Loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("C:/Users/krish/OneDrive/Desktop/MACHINE LEARNING/Image Classification/server/artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open("C:/Users/krish/OneDrive/Desktop/MACHINE LEARNING/Image Classification/server/artifacts/saved_model.pkl", 'rb') as f:
            __model = joblib.load(f)
    print("Loading saved artifacts...done")

def get_cv2_image_from_base64_string(b64str):
    if ',' in b64str:
        encoded_data = b64str.split(',')[1]
    else:
        encoded_data = b64str
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    if image_path:
        print(f"Reading image from path: {image_path}")
        img = cv2.imread(image_path)
    else:
        print("Reading image from base64 data")
        img = get_cv2_image_from_base64_string(image_base64_data)

    if img is None:
        print("Failed to decode image")
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y: y + h, x: x + w]
        roi_color = img[y: y + h, x: x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)
    if not cropped_faces:
        print("No faces with 2 eyes detected")
    return cropped_faces

def get_b64_test_image_for_virat():
    with open("C:/Users/krish/OneDrive/Desktop/MACHINE LEARNING/Image Classification/server/base64.txt") as f:
        return f.read()

if __name__ == '__main__':
    load_saved_artifacts()
    thiago_path = "Image Classification/model/Pictures/messi_son.jpeg"
    result = classify_image(None, thiago_path)
    print("Final result:", result)
