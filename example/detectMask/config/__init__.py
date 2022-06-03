import  torch

# 运行设备选择
RUN_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# RUN_DEVICE = torch.device('cpu')

# 学习率
# Note - 随机初始模型权重训练，就设置大一点（例如，0.0005）。有初始模型权重训练，设置小一些
LEARN_RATE = 0.00005
# 衰减延迟
DELAY_RATE = 0.8
# 训练次数
EPOCHS = 100

# 一个batch包含样本量
# Note - 设置太大了，会爆显存。笔记本 2070 8g ，一次训练 8 张，显存占用 75% (￣_￣|||)
BATCH_SIEZE = 8
# 待检测目标的种类数
NUM_CLASSFICATIONS = 2
# 待检测目标的名称
OBJECT_CLASSFICATIONS = {'no_mask':0,'have_mask':1}
# 输入网络模型的图片尺寸 宽 x 高
INPUT_SIZE = (416,416)
# 一个单元格拥有多少个的 anchor box
NUM_CELL_ANCHOR = 3
# 预先设定的 anchor box 基于 INPUT_SIZE 尺寸，[[ anchor_w,anchor_h ]]
BOX_ANCHORS = torch.tensor([[[10,13],[16,30],[33,23]],
                            [[30,61],[62,45],[59,119]],
                            [[116,90],[156,198],[373,326]]],device=RUN_DEVICE) 

# 大于该阈值的预测框，不参与置信度损失计算
THRESH_IGNORE = 0.85
# 大于该阈值才认为 ground true box 与 anchor box 对应
# Note - 设置大一点，能提高一些 precision 但是会降低 recall
THRESH_GTBOX_ANCHOR_IOU = 0.25
# 位置损失的权重
LAMBDA_COORD = 50
# 置信度损失权重
# Note - 当 precision 与 recall 较低时，需要设置大一些，不然找不到目标
LAMBDA_CONF = 5000
# 分类损失的权重
LAMBDA_CLASS = 500

# 是否存在待检测目标的阈值
THRESH_OBJ = 0.6
# 极大值抑制阈值
# Note - 小于该阈值的预测框会保留，大于该值的预测框会被剔除
THRESH_NMS = 0.4
# score 过滤阈值
THRESH_SCORE = 0.6
# 预测框最小尺寸限制
THRESH_BOX_MIN_SIZE = 8
# 每个分类，最多有多少个预测框进行极大值抑制的筛选
THRESH_BOX_MAX_NUM = 100
# ap 计算时的阈值
THRESH_AP_IOU = 0.5


# 根据待检测目标的名称的标记值，将 OBJECT_CLASSFICATIONS 转为数组
# Note - 自动转换，不用管
def getObjectNames(labels):
    names = [''] * len(labels)
    for key in labels :
        names[ labels[key] ] = key
    return names
NAME_CLASSFICATIONS = getObjectNames(OBJECT_CLASSFICATIONS)
