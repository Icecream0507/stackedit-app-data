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
2. 获取到图片，设计算法获取白线中线位置。
```python
def is_white(point):#判断是否为白色
	if point[1] < 20 and point[2] > 230:
		return 1
	else:
		return 0

def getmid(hsv):#获取白线中线位置，用20个高度上的白线取平均
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
3. 由白线位置和设定值算出小车理论运动速度，使用PD算法控制小车循迹行驶。
```python
try:
last_dmid = 0
dmid = 0
	while True:
		mid = getmid(HSV_frame)
		kp = 2
		kd = 1
		#cv2.imshow("frame",HSV_frame)
		last_mid = dmid
		dmid = 81 - mid
		d = dmid - last_mid
		w = kp * dmid + kd * d
		x_speed = 80
		y_speed = 0
		if(abs(dmid) >= 18):#线性函数调速
			x_speed /= 5
		if(abs(dmid) <= 5):
			x_speed *= -0.125*abs(2*dmid) + 2.25
		#print(mid,"|",dmid)
		car.set_speed(x_speed, y_speed, w)
```
### 总结：
在该任务中，我们通过HSV图像识别、平均中线获取、PD参数控制、以及线性函数调速实现了小车循白线功能，经过测试以及验证，小车运行总体流畅丝滑。达到了预期目标。

## 任务二：双线循迹
### 目标：使小车在白色线和黄色线中间循迹行驶

	
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTExMDQ0ODU1NDQsLTIxMzM2NTM2NzYsMT
I1NTI3MjQ3MywtNTA4NzQ3MzQxLC0yMDg4NzQ2NjEyLDE0NzI0
MjYzNzVdfQ==
-->