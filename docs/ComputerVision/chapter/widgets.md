# 界面控件


<a href="https://github.com/spite-triangle/artificial_intelligence/tree/master/example/computerVision/widget" class="jump_link"> 本章测试程序 </a>

# TrackBar

<p style="text-align:center;"><img src="../../image/computerVision/trackbar.jpg" width="75%" align="middle" /></p>

```python
import cv2
# trackbar 改变时的回调函数
def onTrackbarChange(value):
    print(value)

# 创建界面
cv2.namedWindow('trackbar',cv2.WINDOW_NORMAL)
cv2.resizeWindow('trackbar',width=640,height=360)

# 创建trackbar
# createTrackbar(trackbarName, windowName, defaultValue, maxValue, onChangeCallback) -> None
cv2.createTrackbar('bar','trackbar',0,255,onTrackbarChange)

# 读取trackbar 的值
# getTrackbarPos(trackbarname, windowName) -> trackbarValue
value = cv2.getTrackbarPos('bar','trackbar')
print(value)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

# 几何图形

## 直线、圆形、矩形

```python
# %% 画直线、圆形、矩形
import cv2
import numpy as np

# 创建窗口
cv2.namedWindow('draw_shape',cv2.WINDOW_NORMAL)
cv2.resizeWindow('draw_shape ',width=640,height=360)

# 必须先有一张背景图，用来当画布
canvas = np.zeros(shape=(360,640,3),dtype=np.uint8)
canvas[:,:] = [255,255,0]

# 矩形
# rectangle(canvas:img, pt1:tuple, pt2:tuple, color[, thickness[, lineType[, shift]]]) -> img
imgRect = cv2.rectangle(canvas,(20,40),(100,100),(255,0,0),3)

# 圆形
# circle(canvas:img, center:tuple:, radius, color[, thickness[, lineType[, shift]]]) -> img
imgCircle = cv2.circle(canvas,(200,60),50,(0,0,255),3,16)

# 直线
# img = cv2.line()
# line(canvas:img, pt1:tuple, pt2:tuple, color[, thickness[, lineType[, shift]]]) -> img
imgLine2 = cv2.line(canvas,(20,150),(200,170),(255,0,0),20,2)
imgLine8 = cv2.line(canvas,(20,200),(200,220),(255,0,0),20,8)
imgLine16 = cv2.line(canvas,(20,250),(200,270),(255,0,0),20,16)

cv2.imshow('draw_shape',canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

- **画布：** 绘制形状时，需要首先建立一张背景图片充当画布，并且该画布在绘制完图像后不会改变。**这些 numpy 数组指向的数据实际上都是同一个，这是由 numpy 本身特性所决定的。**
   ```python
    print(canvas.__array_interface__['data'])
    print(imgRect.__array_interface__['data'])
    print(imgCircle.__array_interface__['data'])
    print(imgLine2.__array_interface__['data']) 
   ```

   ```term
   triangle@LEARN:~$ python ./widget.py
    (2355727630464, False)
    (2355727630464, False)
    (2355727630464, False)
    (2355727630464, False)
   ```
- **传参数注意：**
    - `LineType`：该值控制的是「抗锯齿」效果，值越大，线条越光滑。取值通常为 $2^n$
        <p style="text-align:center;"><img src="../../image/computerVision/lineType.jpg" width="25%" align="middle" /></p>

    - `pt`： <span style="color:red;font-weight:bold"> 所有点的坐标表示为 （横向坐标，纵线坐标） </span>
    - `color`：**通道为 (B,G,R)**


## 椭圆

```python
# 椭圆
# axes：长轴、短轴
# angle：椭圆倾斜角度，顺时针
# ArcStartAngle, ArcEndAngle：起始弧长和终止弧长，顺时针
ellipse(img, center:tuple, axes:tuple, angle,
        ArcStartAngle, ArcEndAngle, 
        color[, thickness[, lineType[, shift]]]) -> img
```

## 多边形


<!-- panels:start -->
<!-- div:left-panel -->
```python
# 多边形
# polylines(img, [pts:numpy.ndarray], isClosed, 
#                color[, thickness[, lineType[, shift]]]) -> img
pts1 = np.array([ (20,60),(300,150),(50,300) ])
pts2 = np.array([ (400,60),(300,100) ])
cv2.polylines(canvas,[pts1,pts2],True,(255,0,0))

# 填充的多边形
fillPoly(img, [pts], color[, lineType[, shift[, offset]]]) -> img
```

<!-- div:right-panel -->

<p style="text-align:center;"><img src="../../image/computerVision/polylines.jpg" width="60%" align="middle" /></p>

<!-- panels:end -->

- `[pts]`：可以同时设置多组多边形的点集，并且多组多边形直接不会相互连接。


# 文本

## 英文

<!-- panels:start -->
<!-- div:left-panel -->


```python
# fontfamily：cv2.FONT_ 进行查看
putText(img, text, pos:tuple, fontfamily, fontScale, 
        color[, thickness[, lineType[, bottomLeftOrigin]]]) -> img
```

- `pos`：起始位置为文本的「左下角」

<!-- div:right-panel -->

<p style="text-align:center;"><img src="../../image/computerVision/font_pos.jpg" width="60%" align="middle" /></p>

<!-- panels:end -->


## 中文


<!-- panels:start -->
<!-- div:left-panel -->
```python
import cv2
import numpy as np
from PIL import Image,ImageFont,ImageDraw

# 必须先有一张背景图，用来当画布
canvas = np.zeros(shape=(360,640,3),dtype=np.uint8)
canvas[:,:] = [255,255,0]

# 导入字体
font = ImageFont.truetype('./asset/eva_font.otf',size=35)
# 创建画布
canvasBg = Image.fromarray(canvas)
# 创建画笔
brush = ImageDraw.Draw(canvasBg)
# 写入中文
brush.text(xy=(100,100),text="使徒襲来",font=font,fill=(255,0,0,0))

# 转换图片
canvas = np.array(canvasBg)

```
<!-- div:right-panel -->
<p style="text-align:center;"><img src="../../image/computerVision/zh.jpg" width="75%" align="middle" /></p>
<!-- panels:end -->


# 附录: API传参注释

```python
circle(canvas:img, center:tuple:, radius, 
    color[, thickness[, lineType[, shift]]]) -> img
```

在API中，`[]`代表了可选参数，函数调用时，这些参数不用设置。因此上面的函数必须传递的参数是：`canvas:img, center:tuple:, radius, color`

此外，`[]`还可以嵌套使用，例如`[, thickness[, lineType[, shift]]]`，其含义就是：
1. 在选择传输参数`thickness`之后，`[, lineType[, shift]]` 部分为可选参数，
2. 必须在设置了参数`thickness`的前提下，才能设置参数`[, lineType[, shift]]`

