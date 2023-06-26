import cv2 as cv
import argparse
import numpy as np
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import imageio

parser = argparse.ArgumentParser('run motion detection')
parser.add_argument('-v', '--video_dir', type = str, help ='video directory')
parser.add_argument('-n', '--number_of_video', type = int, help ='number_of_video in video_dir')
args = parser.parse_args()


def check_motion(video_dir, n):
    print(video_dir)
    df = pd.DataFrame(columns =['label', 'video', 'sum_thresh'])
    
    THRESH_list = []
    label_list =[]
    video_list =[]
    frames =[]
    
    
    for i in range(n):
        video =  video_dir + os.listdir(video_dir)[i]
        print(i, video)
        video_list.append(os.listdir(video_dir)[i])


        cap = cv.VideoCapture(video)
        target_brightness = 100  # Adjust the target brightness 
        target_contrast = 1.0   # Adjust the target contrast

        _, first_frame = cap.read()
        
        first_frame = cv.convertScaleAbs(first_frame, alpha=target_contrast, beta=target_brightness)
        
        # apply mask to the center of frame with radius=50 to the first frame
        blank_first = np.zeros(first_frame.shape[:2], dtype ='uint8')
        mask_first = cv.circle(blank_first, (first_frame.shape[1]//2, first_frame.shape[0]//2), 50, 255, -1)
        masked_first = cv.bitwise_and(first_frame, first_frame, mask=mask_first)


        first_bw = cv.cvtColor(masked_first, cv.COLOR_BGR2GRAY)
        first_bw = cv.GaussianBlur(first_bw, (21,21), 0)
        thresh0 =cv.threshold(first_bw, thresh=20, maxval =255, type= cv.THRESH_BINARY)[1]
#         print('first frame: ', thresh0.sum()/(10**6))
        


        while True:
            status, frame = cap.read()
            if status:
                
                frame = cv.convertScaleAbs(frame, alpha=target_contrast, beta=target_brightness)

                #mask (to the center of frame with radius=50) to the i_th  frame
                blank = np.zeros(frame.shape[:2], dtype ='uint8')
                mask = cv.circle(blank, (frame.shape[1]//2, frame.shape[0]//2), 50, 255, -1)
                masked = cv.bitwise_and(frame, frame, mask=mask)

                frame_bw = cv.cvtColor(masked, cv.COLOR_BGR2GRAY)
                frame_bw = cv.GaussianBlur(frame_bw, (5,5), 0)         # blur , to remove unneccesary noise


                diff = cv.absdiff(frame_bw, first_bw)


                # Threshold:  pixel< 20 -> 0 (black) if >20 -> 255 (white)
                thresh =cv.threshold(diff, thresh=20, maxval =255, type= cv.THRESH_BINARY)[1]  

                cv.imshow('thresh', thresh)
                cv.imshow('frame', frame)
                
                frames.append(thresh) # for creating gif result
                
                sum_thresh = thresh.sum()/(10**6)

                cv.waitKey(1)
            else:
                break
                
        THRESH_list.append(sum_thresh)
        
        if sum_thresh >1:      # empirically derived 
            label_list.append('train_in_out')
        else:
            label_list.append('another class')
            
    df.label = label_list
    df.sum_thresh = THRESH_list
    df.video = video_list
    
    print("Saving GIF file")
    with imageio.get_writer("threshold.gif", mode="I") as writer:
        for idx, frame in enumerate(frames):
            if idx%10==0:
                writer.append_data(frame)
        
    cap.release()
    cv.destroyAllWindows()
    return df, writer
    
if __name__ == '__main__':
    print(check_motion(args.video_dir,  args.number_of_video))
    
    
