# 数字识别

# 案例说明
- **目的：** 识别出图片的数字
- **思路：** 首先准备一个数字模板，然后通过模板匹配算法与图片中数字进行对比，进而识别出每个数字。
- <a href="https://github.com/spite-triangle/artificial_intelligence/tree/master/example/computerVision/digitalRecognition" class="jump_link"> 案例项目工程 </a>


# 具体流程

## 模板

<p style="text-align:center;"><img src="../../image/computerVision/digitalTemplate.jpg" width="25%" align="middle" /></p>

- **目标：** 将模板中的数字分别截取出来。
- **思路：**
    1. 图片转灰度；调整图片大小；将图片转为二值图。
    2. 抓取图片中的外轮廓，并生成外轮廓的外接矩形
    3. **根据外接轮廓的`x`坐标值，对外轮廓进行排序，这样就能知道每个数字对应的轮廓**
    4. 知道了数字与轮廓的对应关系，**就又能根据外接矩形坐标，对图片中的数字进行截取**
<p style="text-align:center;"><img src="../../image/computerVision/digitalRoi.jpg" width="25%" align="middle" /></p>

## 目标图像

<p style="text-align:center;"><img src="../../image/computerVision/digitalCard.jpg" width="25%" align="middle" /></p>

- **目标：** 将图片中的数字部分提取出来
- **思路：**
    1. 图片转灰度；调整图片大小；将图片转换为二值图。**用于轮廓提取的图片大小不要设置太大，小一点反而更准确一些。**

    2. 进行形态学、高通滤波（提取轮廓）、低通滤波（去噪）等操作，得到下图。**其目的是为了「将数字区域」全部涂抹成「块」**
        <p style="text-align:center;"><img src="../../image/computerVision/digitalCardMorph.jpg" width="40%" align="middle" /></p>
    3. 对上图进行轮廓检测，检索出「数字区域」的轮廓。**需要对轮廓进行筛选，可以通过轮廓的长宽比、周长、面积等特征。**
        <p style="text-align:center;"><img src="../../image/computerVision/digitalCardContour.jpg" width="40%" align="middle" /></p>
    4. 对筛选的轮廓进行排序，排序方法同模板处理
    5. **通过轮廓外接矩形，对「数字区域」进行截取** 
    6. **有了「数字区域」，又能同模板处理，拆解出一个数字的图片**
    7. 输入的数字图片与模板图片进行「模板匹配」，就检索出各个输入图片的数字