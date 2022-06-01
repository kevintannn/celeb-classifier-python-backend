import cv2
import joblib
import numpy as np
import base64
import json
from wavelet import w2d
import os

__class_name_to_number = {}
__class_number_to_name = {}
__model = None
loaded = False


def get_cv2_image_from_base64_string(b64str):
    """
    credit:
    https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    """
    encoded_data = b64str.split(",")[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return img


def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier(
        "./haarcascades/haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier("./haarcascades/haarcascade_eye.xml")

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x, y, w, h) in faces:
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = img[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)

    return cropped_faces


def class_number_to_name(class_num):
    return __class_number_to_name[class_num]


def classify_image(image_base64_data, file_path=None):
    global loaded
    if loaded == False:
        load_saved_artifacts()
        loaded = True

    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)

    result = []
    for img in imgs:
        scaled_raw_img = cv2.resize(img, (32, 32))

        wavelet_img = w2d(img, "db1", 5)
        scaled_wavelet_img = cv2.resize(wavelet_img, (32, 32))

        # stack vertically
        stacked_img = np.vstack(
            (
                scaled_raw_img.reshape(32 * 32 * 3, 1),
                scaled_wavelet_img.reshape(32 * 32, 1),
            )
        )

        len_image_array = (32 * 32 * 3) + (32 * 32)

        final_image = stacked_img.reshape(1, len_image_array).astype(float)

        result.append(
            {
                "class": class_number_to_name(__model.predict(final_image)[0]),
                "class_probability": np.round(
                    __model.predict_proba(final_image) * 100, 2
                ).tolist(),
                "class_dictionary": __class_name_to_number,
            }
        )

    return result


def get_b64_test_image_for_virat():
    with open("b64.txt") as f:
        return f.read()


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open("saved_model.pkl", "rb") as f:
            __model = joblib.load(f)

    print("loading saved artifacts...done")


# if __name__ == "__main__":
#     load_saved_artifacts()
#     # print(__class_number_to_name)
#     # print(classify_image(get_b64_test_image_for_virat(), None))
#     print(classify_image(None, "./test_images/sharapova1.jpg"))
#     print(classify_image(None, "./test_images/serena1.jpg"))
#     print(classify_image(None, "./test_images/virat1.jpg"))
#     print(classify_image(None, "./test_images/messi1.jpg"))
