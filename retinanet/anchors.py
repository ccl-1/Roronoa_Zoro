import numpy as np
import torch
import torch.nn as nn

# 解析见： https://blog.csdn.net/qq_36251958/article/details/105024133

class Anchors(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, sizes=None, ratios=None, scales=None):
        super(Anchors, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels] #[8, 16, 32, 64, 128]
        if sizes is None: # base_size选择范围
            self.sizes = [2 ** (x + 2) for x in self.pyramid_levels] #[32,64,128,256,512]
        if ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
        
        #对于长宽为（原图大小/8，原图大小/8）的特征图，
        #其特征图上的每个单元格cell对应原图区域上base_size为（32，32）大小的对应区域
        #然后在大小为base_size的正方形框的基础上，对框进行长宽比例调整
        
    def forward(self, image):
        # image为 img_batch=（b,c,h,w）
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        # 不同特征图上的size ,[原图/8，原图/16，原图/32，原图/64，原图/128]
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors         = generate_anchors(base_size=self.sizes[idx], ratios=self.ratios, scales=self.scales)
            shifted_anchors = shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

        all_anchors = np.expand_dims(all_anchors, axis=0)

        if torch.cuda.is_available():
            return torch.from_numpy(all_anchors.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchors.astype(np.float32))

def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    输出 anchors.shape=(9,4),对应面积为 base_size 的9个不同ratios,scales的anchor坐标
    每行为一个anchor:(x1,y1,x2,y2)，中心坐标(x_c,y_c)=(0,0)，故x1,y1为负，x2,y2为正
    """

    if ratios is None:
        ratios = np.array([0.5, 1, 2]) # shape=(3,)

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]) # (3,)

    num_anchors = len(ratios) * len(scales)

    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))

    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
    # np.tile(a,(2)) 等价于 np.tile(a,(1,2))第一个参数为Y轴扩大倍数，第二个为X轴扩大倍数
    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios,按照 ratio 计算长宽
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    # 每个元素重复3次,(0.5,0.5,0.5, 1,1,1, 2,2,2 ).shape=(9,)
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))
    #截止至此，我们得到anchor的数组情况为：[0,0,h,w]

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    # 默认以当前anchor的中心点为坐标原点建立直角坐标系，
    # 求出左上角坐标和右下角坐标，存入当前数组，格式为(x1,y1,x2,y2)。
    # [0,0,h,w] -> [-h/2,-w/2,h/2,w/2]
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T # 0，2 列
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T # 1，3 列

    return anchors

def compute_shape(image_shape, pyramid_levels):
    """Compute shapes based on pyramid levels.

    :param image_shape:
    :param pyramid_levels:
    :return:
    """
    image_shape = np.array(image_shape[:2])
    image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in pyramid_levels]
    return image_shapes


def anchors_for_shape(
    image_shape,
    pyramid_levels=None,
    ratios=None,
    scales=None,
    strides=None,
    sizes=None,
    shapes_callback=None,
):

    image_shapes = compute_shape(image_shape, pyramid_levels)

    # compute anchors over all pyramid levels
    all_anchors = np.zeros((0, 4))
    for idx, p in enumerate(pyramid_levels):
        anchors         = generate_anchors(base_size=sizes[idx], ratios=ratios, scales=scales)
        shifted_anchors = shift(image_shapes[idx], strides[idx], anchors)
        all_anchors     = np.append(all_anchors, shifted_anchors, axis=0)

    return all_anchors


def shift(shape, stride, anchors):
    '''
    generate_anchors 得到的只是当前anchor相对于自身中心点的坐标，
    还要将anchor映射到特征金字塔的每一层的feature map上，以p3为例，p3的尺度为75*75。
    那么映射的思路为：
        1）首先生成75*75的网格坐标
        2）对于每一个网格，将9个anchor分别嵌入进去，得到基于网格的anchor坐标
    
    input:  shapes -> feature map的尺寸
            anchor -> 在input_image上以strides为步长滑动生成的坐标(x2, y1, x2, y2)
    output: shifted_anchors.shape=(shapes[0]*shapes[1]*9, 4)
    '''
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    # np.arang两个参数时，第一个参数为起点，第二个参数为终点，步长取默认值1。
    # 为什么要加0.5，其实我们把75*75看成网格，所求的网格点坐标其实都是每个小网格中心点的坐标，
    # 步长为1，即网格的边长是1，中心点肯定是要加上0.5的。
    
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),shift_x.ravel(), shift_y.ravel()
    )).transpose() # 存储网格中心点坐标（cx,cy,cx,cy） 对应 anchor的相对坐标（x1,y1,x2,y2）

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    # 将anchor中心点换做feature map的网格点，然后将偏移量叠加至上面，则得到anchor到feature map的映射
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors

