import cv2, tkinter as tk, os
from deepface import DeepFace
from tkinter import filedialog
import procesuiranje

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


backend = backends[1] #tu se podesi backend
resized_w, resized_h = 1280, None #dimenzije resizane slike (scaled), jedna dimenzija mora biti None


file_path = "Slike\memelovetriangle.webp" #default slika
def browse_files():
    global file_path
    file_path = filedialog.askopenfilename(initialdir=os.path.join(os.getcwd(), "Slike"), title="Select file")
    label_path.config(text=file_path)
    

def video():
    global resized_w, resized_h
    gumb1.config(state="disabled")
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        orig_w, orig_h = img.shape[1], img.shape[0]  # dimenzije originalne slike
        # solidno rade 1,3,4,7,8  6 radi losije   koma su: 5,0,  brzi su:7,8
        analiza = DeepFace.analyze(img, actions=(
            "emotion", "age", "gender", "race"), detector_backend=backend, enforce_detection=False)
        resize = procesuiranje.ispis(analiza, img, orig_w, orig_h, resized_w, resized_h)
        cv2.imshow("Slika", resize)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    gumb1.config(state="normal")


def slika():
    global resized_w, resized_h
    gumb2.config(state="disabled")
    img = cv2.imread(file_path)
    orig_w, orig_h = img.shape[1], img.shape[0]
    analiza = DeepFace.analyze(img, actions=(
        "emotion", "age", "gender", "race"), detector_backend=backend, enforce_detection=False)
    resize = procesuiranje.ispis(analiza, img, orig_w, orig_h, resized_w, resized_h)
    cv2.imshow("Slika", resize)
    cv2.waitKey(0) & 0xFF == ord("q")
    cv2.destroyAllWindows()
    gumb2.config(state="normal")
    
    

root = tk.Tk()
gumb1 = tk.Button(root, text="Video", command=video, width=15, height=3)
gumb1.grid(row=0, column=0, pady=10, padx=10, sticky="nsew")
gumb2 = tk.Button(root, text="Slika", command=slika, width=15, height=3)
gumb2.grid(row=0, column=1, pady=10, padx=10, sticky="nsew")
gumb3 = tk.Button(root, text="Browse", command=browse_files, width=15)
gumb3.grid(row=1, column=1, pady=10, padx=10, sticky="nsew")
label_path = tk.Label(root, text="Odaberi sliku", anchor="w")
label_path.grid(row=2, column=1, pady=10, padx=10, sticky="w")

for i in range(3):
    root.grid_rowconfigure(i, weight=1)
for i in range(2):
    root.grid_columnconfigure(i, weight=1)

root.mainloop()