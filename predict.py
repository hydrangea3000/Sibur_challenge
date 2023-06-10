from ultralytics import YOLO
import cv2 as cv
import argparse
import os, random
import numpy as np
import math

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

parser = argparse.ArgumentParser('run YOLO trained on custom dataset')
parser.add_argument('-v', '--video_dir', type=str, help='video direction')
parser.add_argument('-m', '--model_dir', type=str, help='model weights direction')
parser.add_argument('-n', '--number_of_video', type = int, help ='number_of_video in video_dir')

args = parser.parse_args()


def run_yolo(model_weights, video_dir, n):  
    
    video =  video_dir + os.listdir(video_dir)[n]
    cap = cv.VideoCapture(video)
    
    model = YOLO(model_weights)
    names = ['carriage', 'bridge']

    colors = [tuple([np.random.randint(0, 255) for _ in range(3)]) for _ in range(len(names))] 
    class_color_dict = {k: v for k, v in zip(names, colors)}
    
    labels =[]
  
    while True:
        status, frame = cap.read()
        if status:
            results = model(frame , stream =True)        
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    #Bbox
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2  = int(x1), int(y1), int(x2), int(y2)

                    #p_c
                    conf = math.ceil((box.conf[0]*100))/100

                    # Class/label
                    label = names[int(box.cls[0])]

                    #draw bbox
                    if conf > 0.45:
                        cv.rectangle(frame, (x1, y1), (x2, y2), class_color_dict[label], thickness=2)
                        cv.putText(frame, f'{label} {conf}', (x1, y1 - 5), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                               class_color_dict[label], 2)
                        labels.append(label)


            cv.imshow('video', frame)
            cv.waitKey(1)
        else:
            break

    cap.release()
    cv.destroyAllWindows()
    return list(set(labels))
    

def check_motion(video_dir, n):
    
    video =  video_dir + os.listdir(video_dir)[n]
    print(video)
    
    cap = cv.VideoCapture(video)
    target_brightness = 100  # Adjust the target brightness 
    target_contrast = 1.0   # Adjust the target contrast

    _, first_frame = cap.read()

    first_frame = cv.convertScaleAbs(first_frame, alpha=target_contrast, beta=target_brightness)

    blank_first = np.zeros(first_frame.shape[:2], dtype ='uint8')
    mask_first = cv.circle(blank_first, (first_frame.shape[1]//2, first_frame.shape[0]//2), 50, 255, -1)
    masked_first = cv.bitwise_and(first_frame, first_frame, mask=mask_first)


    first_bw = cv.cvtColor(masked_first, cv.COLOR_BGR2GRAY)
    first_bw = cv.GaussianBlur(first_bw, (21,21), 0)
    thresh0 =cv.threshold(first_bw, thresh=30, maxval =255, type= cv.THRESH_BINARY)[1]
#     print('first frame: ', thresh0.sum()/(10**6))
    
    while True:
        status, frame = cap.read()
        if status:
        
            frame = cv.convertScaleAbs(frame, alpha=target_contrast, beta=target_brightness)

            #mask
            blank = np.zeros(frame.shape[:2], dtype ='uint8')
            mask = cv.circle(blank, (frame.shape[1]//2, frame.shape[0]//2), 50, 255, -1)
            masked = cv.bitwise_and(frame, frame, mask=mask)

            frame_bw = cv.cvtColor(masked, cv.COLOR_BGR2GRAY)
            frame_bw = cv.GaussianBlur(frame_bw, (5,5), 0)         # blur , to remove unneccesary noise


            diff = cv.absdiff(frame_bw, first_bw)

            # threshold
            thresh =cv.threshold(diff, thresh=20, maxval =255, type= cv.THRESH_BINARY)[1]   # pixel<20 -> 0 (black) if >20 -> 255 (white)

            cv.imshow('thresh', thresh)
            cv.imshow('frame', frame)

            sum_thresh = thresh.sum()/(10**6)
            print(sum_thresh)

            cv.waitKey(1)
        else:
            break        
        
    cap.release()
    cv.destroyAllWindows()
    return sum_thresh

    

def result(model_weights, video_dir, n):
    
    labels = run_yolo(model_weights, video_dir, n)
    print('Labels:', labels)
    
    if 'bridge' in labels:
        return 'bridge_down'
    
    elif 'carriage' in labels  and  not 'bridge' in labels:
        sum_thresh = check_motion(video_dir,  n)
        if sum_thresh > 1:  #how significant is difference
            return 'train_in_out'
        else:
            return 'bridge_up'
        
    elif not 'carriage' in labels:
        return 'no_action'
    
    
if __name__ == '__main__':
    print(result(args.model_dir, args.video_dir, args.number_of_video))
