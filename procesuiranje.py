import cv2
from deepface import DeepFace
import sys
import numpy as np


backends = [
    'opencv',  # brzina
    'ssd',  # brzina
    'dlib',
    'mtcnn',  # za preciznost
    'retinaface',  # za preciznost
    'mediapipe',
    'yolov8',
    'yunet',
    'fastmtcnn',
]

# img scaling, hvala stackoverflow (https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display)
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


cap = cv2.VideoCapture(0)


def okvir_lica(x, y, w, h, img):  # znaci x,y, width, height
    # slika, pocetak, kraj, boja, debljina
    okvir = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 3)


def tekst(text, x, y, w, h, img, orig_w, orig_h, resized_w, resized_h):

    y = y+10  # pomice pocetak teksta dolje da ne izlazi is slike gore, za ostale strane mi nis nije sinulo
    # mijenja velicinu slova s obzirom na velicinu originalne slike
    if (resized_w is not None and orig_w and resized_w > orig_w and orig_w <= 650) or (resized_h is not None and orig_h and resized_h > orig_h and orig_h <= 650):
        font_scale = 0.3
        pomak = 10
        thicc = 1
    else:
        font_scale = 1
        pomak = 20
        thicc = 2

    x_pos, y_pos = x+w+5, y+pomak

    for line in text:
        (text_width, text_height), baseline = cv2.getTextSize(
            line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thicc)
        cv2.rectangle(img, (x_pos, y_pos - text_height), (x_pos + text_width,
                      y_pos + baseline), (0, 0, 0), thickness=cv2.FILLED)  # text background
        cv2.putText(img, line, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thicc)  # text
        y_pos += text_height + pomak


def ispis(analiza, img, orig_w, orig_h, resized_w, resized_h):
    for i in range(len(analiza)):
        face_x, face_y, face_w, face_h = analiza[i]["region"].values()

        # provjere i modifikacije postotaka za sve modele (modeli daju razliciti output)
        # nije najbolje rjesenje al nez bolje trenutno
        if (type(analiza[i]["face_confidence"]) is (float or int)) and analiza[i]["face_confidence"] <= 1:
            # pretvara u lijepe postotke
            confidence = str(
                round(analiza[i]["face_confidence"] * 100, 2)) + "%"
        elif float(analiza[i]["face_confidence"]) <= 1:
            confidence = str(
                round(float(analiza[i]["face_confidence"]) * 100, 2)) + "%"
        elif len(str(analiza[i]["face_confidence"])) >= 4:
            confidence = str(round(analiza[i]["face_confidence"], 2)) + "%"
        else:
            confidence = str(analiza[i]["face_confidence"]) + "%"

        confidence = "F%: " + confidence
        emotion = str(analiza[i]["dominant_emotion"])
        emotion_confidence = str(
            round(analiza[i]["emotion"][emotion], 2)) + "%"
        emotion = "E: " + emotion + " " + emotion_confidence

        race = str(analiza[i]["dominant_race"])
        race_confidence = str(round(analiza[i]["race"][race], 2)) + "%"
        race = "R: " + race + " " + race_confidence

        age = "A: " + str(analiza[i]["age"])

        gender = str(analiza[i]["dominant_gender"])
        gender_confidence = str(round(analiza[i]["gender"][gender], 2)) + "%"
        gender = "G: " + gender + " " + gender_confidence

        text = [confidence, gender, age, race, emotion]
        okvir_lica(face_x, face_y, face_w, face_h, img)
        tekst(text, face_x, face_y, face_w, face_h, img, orig_w, orig_h, resized_w, resized_h)

        resize = ResizeWithAspectRatio(
            img, resized_w, resized_h)  # resizanje slike
    return resize
    