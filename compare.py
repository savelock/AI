import cv2
import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine

# ArcFace+RetinaFace 
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

def get_face_embedding(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)
    if len(faces) == 0:
        raise Exception("얼굴을 찾을 수 없음")
    return faces[0].embedding 

vector = np.load("/Users/bagseonghyeon/Desktop/0421지켜락/web/vectors/hyeon.npy") #(512,)

input_vector = get_face_embedding("/Users/bagseonghyeon/Desktop/0421지켜락/karina.jpeg")
similarity = 1 - cosine(vector, input_vector)
threshold = 0.7

if similarity >= threshold:
    print(f"유사도: {similarity:.4f} → 같은 사람")
else:
    print(f"유사도: {similarity:.4f} → 다른 사람")