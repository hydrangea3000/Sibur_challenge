# Sibur challenge
The task of CV competition was to define what is happening on the video with tank-wagon. There are 4 classes: _'bridge down'_ (maintenance operations, there might be workers on the video), _'bridge up'_ (when the wagon has just came to station or is ready to depart, the wagon is stationary), _'no action'_ (no wagon on the video, but there might be workers), _'train_in_out'_ (the wagon is moving: arriving to or departing from the station).

My Solution Pipeline:
1) Data labeling: 2 labels -  'carriage' (tank wagon) and 'bridge' using `Labelme`

   labeling images in `labelme` returns json format, so we need to convert to YOLO format using `labelme2yolo`
2) Training YOLOv8 Instance Segmenation model on Custom Dataset ('carriage' / 'bridge') in [Colab](Training_segmentation.ipynb)
3) [predict.py](https://github.com/hydrangea3000/Sibur_challenge/blob/main/predict.py) contains  Python script using OpenCV (cv2) and YOLOv8 to run inference on video frames and  motion detection algorithm based on difference between frames.

   - If there is a 'bridge' in results of YOLOv8 then we can define the final class as _'bridge down'_
 
   - If there is a 'carriage' and no 'bridge' - then there are 2 options: _'bridge up'_ or '_train_in_out'_ --> Apply  Motion Detection function
 
   - If there is a significant difference (thresh) -> there is a motion -> _'train_in_out'_, else (no motion) -> '_bridge up'_
 
   - If there is no 'carriage' and no 'bridge' in results -> _'no action'_

An additional complication of the competition was to create action recognition model which works fast on CPU. This can be achieved by converting YOLO to 'onnx' format (which is not availbale at the moment for Segmentation due to 'task=segment' param) or/ and take each _i_th_ frame.
