from mtcnn import MTCNN
import cv2
import matplotlib.pyplot as plt
detector = MTCNN()
#cach anh xa mau ra man hinh- con so-> black box-> mau tren man hinh, ta lm viec voi con so de hien thi nhu y muon, cai hien thi de may tinh lo
img = cv2.imread("vungtau.jpg")
img=cv2.resize(img,(400,500))
detection_faces = detector.detect_faces(img)
#print(detections)
for face in detection_faces:
    print(face)
    #print(face['box'])
    x1,y1=face['box'][0:2]
    w,h=face['box'][2:4]
    #print(x1,y1)
    #print(w,h)
    cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(255,0,255),2)
    keypoints=face['keypoints']
    left_eye=keypoints['left_eye']
    right_eye=keypoints['right_eye']
    nose=keypoints['nose']
    mouth_left=keypoints['mouth_left']
    mouth_right=keypoints['mouth_right']
    cv2.circle(img, (left_eye), 2, (0,155,255), 2)
    cv2.circle(img, (right_eye), 2, (0,155,255), 2)
    cv2.circle(img, (nose), 2, (0,155,255), 2)
    cv2.circle(img, (mouth_left), 2, (0,155,255), 2)
    cv2.circle(img, (mouth_right), 2, (0,155,255), 2)
cv2.imshow('vungtau',img)
cv2.waitKey(0)

# mình dùng model train sẵn đi dự đoán một tập dữ liệu nào đó
# sau đó tính sai số giữa kết quả dự đoán với kết quả thực
cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    detection_faces = detector.detect_faces(img)
    for face in detection_faces:
        x1,y1=face['box'][0:2]
        w,h=face['box'][2:4]
        cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(255,0,255),2)
        #keypoints = face['keypoints']
        #left_eye = keypoints['left_eye']
        #right_eye = keypoints['right_eye']
        #nose = keypoints['nose']
        #mouth_left = keypoints['mouth_left']
        #mouth_right = keypoints['mouth_right']
        #cv2.circle(img, left_eye, 2, (0, 155, 255), 2)
        #cv2.circle(img, right_eye, 2, (0, 155, 255), 2)
        #cv2.circle(img, nose, 2, (0, 155, 255), 2)
        #cv2.circle(img, mouth_left, 2, (0, 155, 255), 2)
        #cv2.circle(img, mouth_right, 2, (0, 155, 255), 2)
    cv2.imshow("video",img)
    if cv2.waitKey(1)& 0xFF==ord('q'):
        break