#ta muon xài yolo trên máy mình nhưng máy ko đủ mạnh để train thì ta cứ train lên colab
#tải yolo về máy mình xog rồi tải file train từ colab về dùng.
from my_model.mtcnn_src import FaceDetectorMTCNN
import cv2
from facenet_pytorch import MTCNN
import streamlit as st
import torch
import numpy as np

import io
import os
import scipy.misc
import six
import time
import glob
from IPython.display import display

from six import BytesIO

from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from SSD_face_mask.object_detection.utils import ops as utils_ops
from SSD_face_mask.object_detection.utils import label_map_util
from SSD_face_mask.object_detection.utils import visualization_utils as vis_util
import time

def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict







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


#SSD_face_mask
#Load model
if detection_model=="ssd_face_mask":
    tf.keras.backend.clear_session()
    model_ssd = tf.saved_model.load(r"C:\Users\TUAN\PycharmProjects\CS406\SSD_face_mask\saved_model")
    category_index = label_map_util.create_category_index_from_labelmap(r"C:\Users\TUAN\PycharmProjects\CS406\SSD_face_mask\label_map.txt", use_display_name=True)




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
        cv2.putText(predict, FPS, (7, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    elif detection_model=='ssd_face_mask':
        #start=time.time()
        output_dict = run_inference_for_single_image(model_ssd, frame)
        vis_util.visualize_boxes_and_labels_on_image_array(

            frame,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=3)
        #finish=time.time()
        #FPS=1/(finish-start)
        #FPS=int(FPS)
        #FPS=str(FPS)
        #cv2.putText(frame, FPS, (7, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        #count class
        upto_idx = 0
        for i in range(len(output_dict['detection_scores'])):
            if output_dict['detection_scores'][i] < 0.5:
                upto_idx = i
                break
        objs, counts = np.unique(output_dict['detection_classes'][:upto_idx], return_counts=True)
        lst=[]
        for i in range(len(objs)):
            sub_lst=[objs[i],counts[i]]
            lst.append(sub_lst)
        pos=0
        for couple in lst:
            if couple[0]==1:
                text="with mask:"
            elif couple[0]==2:
                text="without mask:"
            else:
                text="mask_weared_incorrect:"
            cv2.putText(frame, text + str(couple[1]), (50, 50+pos), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 150, 0), 1)
            pos+=20



        #print("Found %d objects." % upto_idx)
        #print(objs, counts)
        predict=frame



    FRAME_WINDOW.image(predict)
    #st.image(predict)#nếu dùng thế này nó sẽ hiển thị ra chuỗi các ảnh
    #ta muốn nó hiển thị các hình ảnh trên cùng một chỗ theo thời gian để tạo thành 1 vd
    #thì ta chỉ dùng 1 địa chỉ để nó hiển thị, mỗi lần st.image nó tạo 1 vùng hiển thị khác nhau, ta chỉ tạo 1 vùng để nó hiển thị lên đó
    #https://docs.streamlit.io/library/api-reference/media
