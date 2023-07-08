# Sibur challenge 2023
[link to competition](https://platform.aitoday.ru/event/9)

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
4) [motion_detection.py]() - _optional_ Python script to show in details how motion detection algo works on 12 random videos from trainig dataset (videos_test)

-> output: [threshold.gif]() and dataframe:

|label|video|sum_thresh|
|-----|-----|----------|
|another class|0cbedb20b827e285.mp4|0.677280|
|train_in_out|1722443d154fd587.mp4|1.462680|
|another class|29edd98ae0bc4e08.mp4|0.773415|
|train_in_out|2ce08918cccbf9d0.mp4|2.244765|
|another class|2ed1d2e5c06d5472.mp4|0.713235|
|another class|42b9e817c0190acd.mp4|0.691305|
|train_in_out|4fca1d1c23743300.mp4|1.003680|
|another class|5bb806639d163766.mp4|0.746130|
|another class|84440a85bdcda906.mp4|0.726495|
|another class|d28e45d3c71083f2.mp4|0.656370|
|another class|f9397b79bb605494.mp4|0.868020|
|train_in_out|fa37a03b72bd12b0.mp4|2.102220|
train_in_out|fbdf51391f396341.mp4|1.698555|
         

   
   

The limitation of the competition was to create action recognition model which works fast on CPU (restricted inference time for 202 test videos is 18 minutes). This can be achieved by converting YOLO to 'onnx' format (which is not availbale at the moment for Segmentation due to 'task=segment' param) or/ and take each _i_th_ frame.
