# 一.实验概述
## 任务一：单线循迹
### 目标：使小车沿着固定的白色单线行进。
实验步骤以及代码摘要：
1. 小车由摄像头获取图片信息。
```python
frame = videostream.read()
frame = cv2.resize(frame, None, fx = 0.25, fy = 0.25, interpolation= cv2.INTER_NEAREST) 
HSV_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
```
3. 获取到图片设计算法判断白线位置。
4. 由白线位置和设定值算出小车理论运动速度，控制小车循迹行驶。

	
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTE2MTY3MzY2MywxMjU1MjcyNDczLC01MD
g3NDczNDEsLTIwODg3NDY2MTIsMTQ3MjQyNjM3NV19
-->