
# 一.实验概述

注：实验场地如图

![court](/imgs/2024-12-08/xiRnVF3jh4jGsha8.jpeg)

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

实验步骤以及代码摘要：

1. 获取白、黄线，设计平均算法获取合理中线位置。

```python

def is_white(point):#白色判断

if point[1] < 20 and point[2] > 230:

return 1

else:

return 0

def is_yellow(point):#黄色判断

if point[0] > 26 and point[0] < 34 and point[1] >43 and point[2] > 46:

return 1

else:

return 0

def getmid(hsv):#中线获取

midline = [81]

for y in range(80, 100):

white_x = []

yellow_x = []

for x in range(0,160):

if is_white(hsv[y][x]):

if x < 81:

white_x.append(x)

hsv[y][x] = (0, 0, 0)

if is_yellow(hsv[y][x]):

yellow_x.append(x)

hsv[y][x] = (0 ,0 ,0)

if(len(white_x) == 0 or len(yellow_x) == 0):

pass

else:

#midline.append(white_x[-1])

midline.append((white_x[-1] + yellow_x[-1])/2)

hsv[y][int(midline[-1])] = (0, 0, 0)

return sum(midline)/len(midline)

```

3. 与任务一类似的PD控制小车循迹。

代码略

  

### 总结：

在任务二中，我们增加了黄线获取函数，更改PD参数使得小车双线循迹更加稳定，其余步骤与任务一类似。达到预期效果

  

## 任务三：搭建神经网络实现标志路牌识别

### 目标：通过简单神经网络识别路牌并按照路牌指示运行

实现步骤以及代码摘要：

1. 神经网络模型搭建。

```python

# 数据预处理

transform = ......

# 定义 CNN 模型

class CNN(nn.Module):

def __init__(self, num_classes):

super(CNN, self).__init__()

self.conv = nn.Sequential(

nn.Conv2d(1, 16, 3, padding=1),nn.Conv2d(16, 16, 5),nn.ReLU(),

nn.MaxPool2d(2, stride=2),nn.Dropout(0.3),nn.Conv2d(16, 32, 5),

nn.ReLU(),nn.MaxPool2d(2, stride=2),nn.Dropout(0.3)

)

self.fc = nn.Sequential(

nn.Linear(32 * 4 * 4, 100),nn.ReLU(),

nn.Linear(100, num_classes) # 输出类别数量，这里是4

)

self.initialize_weights()

  

def initialize_weights(self):

for m in self.modules():

if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):

nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

if m.bias is not None:

nn.init.constant_(m.bias, 0)

  

def forward(self, x):

x = self.conv(x)

x = x.view(-1, 32 * 4 * 4)

x = self.fc(x)

return x

```

2. 小车端获取蓝色路牌区域，实现数据采集与标注。

```python

import cv2

import numpy as np

# 定义HSV空间中蓝色的范围

lower_blue = np.array([110, 50, 50])

upper_blue = np.array([130, 255, 255])

# 打开摄像头

cap = cv2.VideoCapture(0)

i = 1

while True:

ret, frame = cap.read()

if not ret:

print("Failed to grab frame")

break

# 将图像转换到HSV空间

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# 创建一个掩码，只保留蓝色的部分

mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 使用掩码提取蓝色区域

blue_only = cv2.bitwise_and(frame, frame, mask=mask)

# 找到蓝色区域的轮廓

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,

cv2.CHAIN_APPROX_SIMPLE)

# 遍历轮廓，找到最大的轮廓（假设最大的轮廓是路牌）

if contours:

largest_contour = max(contours, key=cv2.contourArea)

x, y, w, h = cv2.boundingRect(largest_contour)

# 在原图像上绘制矩形框

cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 提取路牌区域

plate = frame[y:y + h, x:x + w]

# 显示原图像和框出的路牌

cv2.imshow('Frame', frame)

cv2.imshow('License Plate', plate)

# 保存路牌图像

if cv2.waitKey(1) & 1 == ord('c'):

cv2.imwrite(f'./left/license_plate{i}.png', plate)

# cv2.imwrite(f'./right/license_plate{i}.png', plate)

# cv2.imwrite(f'./straight/license_plate{i}.png', plate)

# cv2.imwrite(f'./stop/license_plate{i}.png', plate)

i += 1

# 按'q'退出循环

if cv2.waitKey(1) & 1 == ord('q'):

break

```

3. 电脑端模型训练、测试。

```python

# 训练函数

def train_model(model, train_loader, criterion, optimizer, num_epochs, device):

......

# 保存训练好的模型

torch.save(model.state_dict(), 'traffic_sign_model.pth')

  

# 测试函数

def test_model(model, test_loader, device):

model.eval()

......

print(f'Test Accuracy: {accuracy:.2f}%')

  

def main():

# 加载数据

train_dataset = ImageFolder('data/train', transform=transform)

......

# 训练模型

train_model(model, train_loader, criterion, optimizer, num_epochs, device)

# 测试模型

test_model(model, test_loader, device)

if __name__ == '__main__':

multiprocessing.freeze_support()

main()

  

```

4. 在树莓派上部署训练好的模型，识别标志牌并输出结果进行测试。

```python

# 加载模型

num_classes = 4 # 类别数量

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN(num_classes).to(device)

model.load_state_dict(torch.load('traffic_sign_model.pth'))

model.eval()

# 定义数据预处理

transform = ......

# 类别标签

class_labels = ['left', 'park', 'right', 'straight']

# 打开摄像头

cap = cv2.VideoCapture(0)

while True:

ret, frame = cap.read()

if not ret:

break

# 蓝色的 HSV 值范围

lower_blue = np.array([100, 150, 50])

upper_blue = np.array([140, 255, 255])

# 创建蓝色掩码

mask = cv2.inRange(hsv, lower_blue, upper_blue)

masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

# 寻找轮廓来确定蓝色区域

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,

cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:

# 计算轮廓的边界框

x, y, w, h = cv2.boundingRect(cnt)

# 设置最小尺寸过滤噪声

if w > 30 and h > 30:

# 画出边界框

cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# 提取蓝色区域并预处理为模型输入格式

sign_img = frame[y:y+h, x:x+w]

......

# 将图像输入模型并预测类别

with torch.no_grad():

......

predicted_class = class_labels[predicted.item()]

confidence = probabilities[0][predicted.item()].item()

# 显示预测结果

cv2.putText(frame, f"{predicted_class}: {confidence:.2f}", (x, y - 10),

cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2

# 显示处理后的帧

cv2.imshow('Traffic Sign Recognition', frame)

```

5. 小车运动逻辑设计。

```python

with torch.no_grad():

......

predicted_class = class_labels[predicted.item()]

......

# 根据识别结果设计⾏为

# 识别结果为 straight ，按单线巡迹逻辑前进；

if predicted_class == 'straight':

car.set_speed(x_speed, y_speed, w)

# 识别结果为 left ，左转1s；

elif predicted_class == 'left':

car.set_speed(0, 0, 40)

time.sleep(1)

# 识别结果为 right ，右转1s；

elif predicted_class == 'right':

car.set_speed(0, 0, -40)

time.sleep(1)

# 识别结果为 park ，⽴即停⻋并退出循环，程序终⽌；

elif predicted_class == 'park':

car.set_speed(0, 0, 0)

break

# 未识别到指⽰牌，按单线巡迹逻辑前进。

else:

car.set_speed(x_speed, y_speed, w)

```

  

### 总结：

任务三中我们通过搭建模型、采集数据、训练部署、逻辑执行几个步骤完成了小车识别标志牌和合理运行的目标。这是整个实验中最复杂的部分，最终效果不错，也达到预期。

  

# 二.重难点分析与解决方案

## 在实验中遇到了诸多大大小小的问题，在此部分摘录出重难点并分析解决方案。

### 问题一：巡线时候车身控制不稳定，转弯时候角速度不合理。

解决策略：扩展算法为PD算法，调节参数；尝试更换线性函数为非线性；设置转弯判断逻辑以及对应调速。

```python

last_mid = dmid # PD算法控制，引入差分反馈量

dmid = 81 - mid

d = dmid - last_mid

w = kp * dmid + kd * d

  

if(abs(dmid) >= 18): # 线性函数调速

x_speed /= 5

if(abs(dmid) <= 5):

x_speed *= -0.125*abs(2*dmid) + 2.25

```

  

### 问题二：无法得到代码具体运算变量值和数据可视化结果，调试困难。

在HSV图像上修改对应像素点处颜色，直接赋值之后输出图像，可以实时观测程序判断结果以及运算变量大致范围，在之后的标志牌结果识别沿用了这一方案使得调试参数更加直观有效。

  

### 问题三：HSV范围不精确导致其他颜色误判。

解决策略：输出所判断颜色HSV，确定更精确的范围，使条件较为苛刻。

```python

print(HSV_frame[60][80])#实例代码

```

### 问题四：训练好模型后，部署至小车上之后无法识别。

解决策略：通过电脑端验证代码，小车上测试。原因是树莓派上的系统是32位的，默认的数据精度是 float单精度浮点数 ，⽽在本地的训练环境是64位的系

统，默认的数据精度都是 double双精度浮点数 ，因此直接在树莓派上进⾏推理会存在数据精度的差异，导致推理效果不⼀致。可以通过``` .double()``` ⽅法将⽹络精度和⽹络的输⼊都设置成 double 类型。

```python

model.double()

outputs = model(sign_img.double())

```

  

### 问题五：耦合识别与巡线代码后小车巡线行为异常。

解决策略：通过输出变量，发现是由于图像输入大小与之前不一致，通过copy一张新的图像，resize之后输入中线提取函数解决。

  
  
  

### 问题六：小车识别路牌时误识别多个，运行逻辑错误。

解决策略：通过引入比较变量以及大小阈值，使得小车在一定视野下获取最大路牌，并且在该路牌足够大时做出行动。

```python

max_cnt = 0

max_size = 0

x, y, w1, h = 0, 0, 0, 0

for cnt in contours:

x, y, w1, h = cv2.boundingRect(cnt)

if w1 * h > max_size:

max_cnt = cnt

max_size = w1 * h

if max_cnt != 0:

x, y, w1, h = cv2.boundingRect(max_cnt)

```

  

### 总结：面对实验过程的种种问题，我们认真思考，积极求解，最终过五关斩六将解决了所有的问题。

  

# 三.本实验的收获

1. 本次实验是与图像处理，运动控制相关的麦克纳姆轮小车实验。在实验中我初步接触了OpenCV的图像处理相关知识，运用了HSV色域、PD算法控制等初等算法，以及简单的CNN网络模型，对控制领域以及简单的神经网络有了初步的了解与认识。

  

2. 实验过程中我和队友通力协作，相互配合，共同解决了诸多问题，体会到了团队协作的力量。我也从他那里学到了MARKDOWN的基础语法。

  

3. 整体上，这次实验是理论与实践相结合的具体项目，不仅让我收获了知识，体悟了团队精神，还培养了我的工程能力，是一次精彩的实践之旅。

  

最后附上一张我们的合影。深深感谢我的队友、课程老师和助教！

![帅照](/imgs/2024-12-08/MyUTInHv6LTNr7cj.jpeg)

<!--stackedit_data:
eyJoaXN0b3J5IjpbNDQyOTg1OCwtMTAwNDIxMzI1MiwxMjc1NT
A3MzUwLDg5Mzk4ODU4NCwtMjEzMzY1MzY3NiwxMjU1MjcyNDcz
LC01MDg3NDczNDEsLTIwODg3NDY2MTIsMTQ3MjQyNjM3NV19
-->