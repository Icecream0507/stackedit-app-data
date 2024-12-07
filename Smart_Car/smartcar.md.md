# 一.实验概述
## 任务一：单线循迹
### 目标：使小车沿着固定的白色单线行进。
实验步骤以及代码摘要：
1. 小车由摄像头获取图片信息。
```python
frame = videostream.read()
frame = cv2.resize(frame, None, fx = 0.25, fy = 0.25,
interpolation= cv2.INTER_NEAREST) 
HSV_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#转为HSV色域
```
2. 获取到图片，设计算法获取白线位置。
```python
def is_white(point):#判断是否为白色
	if point[1] < 20 and point[2] > 230:
		return 1
	else:
		return 0

def getmid(hsv):#获取白线中线位置，用20ge
	midline = []
	for y in range(80, 100):
		white_x = [81]
	for x in range(20,140):
		if is_white(hsv[y][x]):
			white_x.append(x)
			hsv[y][x] = (0, 0, 0)#将检测到的白色图像描黑便于输出调试
		if(len(white_x) == 0):
			pass
	else:
		midline.append(sum(white_x)/len(white_x))
		hsv[y][int(midline[-1])] = (0, 0, 0)
	return sum(midline)/len(midline)
```
3. 由白线位置和设定值算出小车理论运动速度，控制小车循迹行驶。
```python

```
	
<!--stackedit_data:
eyJoaXN0b3J5IjpbODIwNjMyNDQ2LC0yMTMzNjUzNjc2LDEyNT
UyNzI0NzMsLTUwODc0NzM0MSwtMjA4ODc0NjYxMiwxNDcyNDI2
Mzc1XX0=
-->