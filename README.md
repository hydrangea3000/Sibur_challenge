# Sibur challenge
The task of CV competition was to define what is happening on the video with tank-wagon. There are 4 classes: _'bridge down'_ (maintenance operations, there might be workers on the video), _'bridge up'_ (when the wagon has just came to station or is ready to depart, the wagon is stationary), _'no action'_ (no wagon on the video, but there might be workers), _'train_in_out'_ (the wagon is moving: arriving to or departing from the station).

My Solution conisits of:
1) Label data: 2 labels -  'carriage' (tank wagon) and 'bridge' using `Labelme`

   labeling images in `labelme` returns json format, so we need to convert to YOLO format using `labelme2yolo`
2) Training YOLOv8 Instance Segmenation model on Custom Dataset ('carriage' / 'bridge') in [Colab](Training_segmentation.ipynb)
3) If there is a 'bridge' in labels then we can define the final class as 'bridge down'
4) If there ia s 'carriage' and no 'bridge' - then there are 2 options: 'bridge up' or 'train_in_out' --> Apply  Motion Detection function based on difference bwtween frames

   If there is a significant difference (thresh) -> there is motion -> 'train_in_out', else no motion -> 'bridge up'
5) If there is no 'carriage' and no 'bridge' in labels -> 'no action'

An additional complication of the competition was to create action recognition model which works fast on CPU. This can be achieved by converting YOLO to 'onnx' format (which is not availbale at the moment for Segmentation) or/ and take each _i_th_ frame.
