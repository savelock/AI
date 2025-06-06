import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis

# FaceAnalysis 객체 생성 - buffalo_l 모델 사용 (RetinaFace + ArcFace)
# CPU를 사용하여 얼굴 검출 및 특징 추출을 수행
# app = FaceAnalysis(name='retinaface_r50_v1', providers=["CPUExecutionProvider"])
app = FaceAnalysis(name='buffalo_l', providers=["CPUExecutionProvider"])

# 모델 초기화 및 준비
# ctx_id=0은 첫 번째 GPU 사용을 의미하지만, CPU provider를 지정했으므로 CPU에서 실행됨
app.prepare(ctx_id=0)

frame = cv2.imread("/Users/bagseonghyeon/Desktop/0421지켜락/test.jpg")

faces = app.get(frame) 

for face in faces:
    x1, y1, x2, y2 = face.bbox.astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Face Detection', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()