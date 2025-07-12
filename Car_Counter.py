from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *  # from sort import everything
import numpy as np

cap = cv2.VideoCapture(r"C:\Users\ooo\Downloads\4K Video of Highway Traffic_(1080P_HD).mp4")

model = YOLO(r"G:\PYcharm Projects\ML\Yolo-Weights\yolov8l.pt")
'''print(model.names)
   {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
    9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
    24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
    31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
    37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
    52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
    60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
    74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
'''

mask = cv2.imread("mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)

'''Limits for line [x1, y1, x2, y2]'''
limits = [20, 300, 650, 300]

vehicleCountList = []

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    imgRegion = cv2.bitwise_and(img, mask)
    '''It compares the corresponding bits of each color channel (R, G, B or grayscale).
       If both bits are 1, the result bit is 1. Otherwise, it's 0.'''
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))  # [0 for _ in range()]

    for r in results:
        boxes = r.boxes
        for box in boxes:
            '''print(box) - each box will give this type of data
                cls: tensor([56.])
                conf: tensor([0.3418])
                data: tensor([[9.0696e+02, 1.6930e+02, 1.2580e+03, 7.0749e+02, 3.4178e-01, 5.6000e+01]])
                id: None
                is_track: False
                orig_shape: (720, 1280)
                shape: torch.Size([1, 6])
                xywh: tensor([[1082.4937,  438.3954,  351.0730,  538.1976]])
                xywhn: tensor([[0.8457, 0.6089, 0.2743, 0.7475]])
                xyxy: tensor([[ 906.9572,  169.2966, 1258.0302,  707.4941]])
                xyxyn: tensor([[0.7086, 0.2351, 0.9828, 0.9826]])
                0.35'''

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            '''For displaying a fancy box using cvzone'''
            w, h = x2-x1, y2-y1
            # cvzone.cornerRect(img, bbox=(x1, y1, w, h), l=30)

            '''for displaying confidence value'''
            confidence = box.conf[0]
            confidence = math.ceil(confidence*100) / 100
            # displaying next

            '''For displaying class name'''
            classValue = int(box.cls[0])
            className = model.names[classValue]
            if ((className == 'car' or className == 'bus' or className == 'truck' or className == 'motorcycle') and
                    confidence >= 0.3):
                cvzone.cornerRect(img, bbox=(x1, y1, w, h), l=5, rt=5)
                cvzone.putTextRect(img, text=f'{className}:{confidence}%',
                                   pos=(max(0, x1), max(35, y1)), scale=1.3, thickness=2, offset=3)

                currentArray = np.array([x1, y1, x2, y2, confidence])
                detections = np.vstack((detections, currentArray))
                '''>>> a = np.array([1, 2, 3])
                   >>> b = np.array([4, 5, 6])
                   >>> np.vstack((a,b))
                   array([[1, 2, 3],
                          [4, 5, 6]])'''

    resultsTracker = tracker.update(detections)

    line = cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), color=(255, 0, 0), thickness=5)

    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        print(result)

        '''Finding the center of each detection : mid point formula'''
        cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
        '''Lets display this center for each vehicle'''
        cv2.circle(img, (cx, cy), radius=3, color=(255, 0, 0), thickness=cv2.FILLED)

        if (vehicleCountList.count(Id) == 0) and limits[0] < cx < limits[2] and limits[1]-15 < cy < limits[3]+15:
            '''we need to make sure that each id is only once and not more than once'''
            vehicleCountList.append(Id)
            line = cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), color=(0, 255, 0), thickness=5)

    cvzone.putTextRect(img, text=f'Vehicle Count:{len(vehicleCountList)}',
                       pos=(50, 50), scale=3, thickness=3, offset=3)
    cv2.imshow('Live', img)
    # cv2.imshow("Region", imgRegion)
    cv2.waitKey(1)
