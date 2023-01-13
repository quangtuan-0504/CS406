from my_model.mtcnn_src import FaceDetectorMTCNN
import cv2
from facenet_pytorch import MTCNN
import streamlit as st
import torch
import numpy as np



import time



camera = cv2.VideoCapture(0)

st.title("Result")

detection_model = st.sidebar.selectbox('Choose detection model', ['none', 'gray', 'mtcnn','face_mask_yolov5','ssd_face_mask'])
detection_thresh = st.sidebar.slider(label='Detection theshold', min_value=0.0, max_value=1.0)
st.sidebar.write('')
st.sidebar.write('')

gender_model = st.sidebar.selectbox('Choose gender recognize model', ['none', 'gray', 'mtcnn'])
gender_thresh = st.sidebar.slider(label='Gender recognize theshold', min_value=0.0, max_value=1.0)

st.sidebar.write('')
st.sidebar.write('')
run = st.sidebar.checkbox('Apply')
st.sidebar.write('Please check "Apply" button is empty before choose new effect!')



# load model khi nào chọn mới load model
#MTCNN
if detection_model=="mtcnn":
    mtcnn = MTCNN()
    fcd = FaceDetectorMTCNN(mtcnn)


#YOLO_face_mask
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
#phỉa có wf mới load dc model nha
if detection_model=="face_mask_yolov5":
    model = torch.hub.load('ultralytics/yolov5', 'custom',path=r'C:\Users\TUAN\PycharmProjects\CS406\Yolov5\yolov5\runs\train\exp6\weights\best.pt')
    model.agnostic = True





FRAME_WINDOW = st.image([])
# list img thì các hình trong list sẽ hiện lên cùng 1 khung theo tg.
while run:
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)# lật ảnh
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if detection_model == "none":
        predict = frame

    elif detection_model == 'gray':
        predict = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    elif detection_model == "canny":
        t_lower = 50  # Lower Threshold
        t_upper = 150  # Upper threshold
        predict = cv2.Canny(frame, t_lower, t_upper)

    elif detection_model == "mtcnn":
        predict = fcd.run(frame)
    elif detection_model=="face_mask_yolov5":
        #trong này muốn chạy yolov5 dự đoán các frame cho ra kết quả là ảnh dự đoán lưu dạng tensor
        #np.queeze bỏ những cái trục có 1 phần tử
        start=time.time()
        results = model(frame)
        predict=np.squeeze(results.render())
        finish = time.time()
        FPS=1/(finish-start)
        FPS=int(FPS)
        FPS=str(FPS)
        cv2.putText(predict, 'FPS:'+FPS, (7, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)



    FRAME_WINDOW.image(predict)