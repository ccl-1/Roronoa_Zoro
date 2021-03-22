import numpy as np
import torch
import torch.nn as nn

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class FocalLoss(nn.Module):
    ''' 使用Focal Loss解决前景和背景样本不平衡的问题    
    
    focal loss 具体计算：
    L_fl =  -   alpha  * (1-p)^gamma * log( p )       if y=1
           - (1-alpha) *     p^gamma * log(1-p)       if y=0
    令 p_t =      p   if y=1
                1-p   otherwise,   即 p_t= y*p + (1-y)*(1-p)
    则，L_fl = - alpha * （1-p_t)^gamma * log(p_t)）
    
    '''
    #def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]
        # 转化为 （cx,cy,w,h）
        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights
        
        # 对于batch_size中的每一张图片，做以下处理
        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1] # category_id != -1
            
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            
            #当图片没有目标时，只计算分类loss，不计算box位置loss（即回归loss=0），所有anchor都是负样本
            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())
                    
                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor 
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma) 

                    bce = -(torch.log(1.0 - classification)) 

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())
                    
                continue


            # 计算所有anchor与真实框的IOU大小，并且找到所有anchor IOU最大的真实框的索引以及该IOU大小
            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations
            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # shape= (num_anchors x 1)

            #import pdb
            #pdb.set_trace()

            # -------------------- 计算 【分类损失】 ------------------------------
            targets = torch.ones(classification.shape) * -1

            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0    # IOU<0.4为负样本，bool,（anchor_nums,1）
            positive_indices = torch.ge(IoU_max, 0.5)   #IOU>0.5为正样本，bool,（anchor_nums,1）
            num_positive_anchors = positive_indices.sum()
            assigned_annotations = bbox_annotation[IoU_argmax, :]    #（anchor_nums,4）

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha
            
            # 注意，这里是利用sigmoid输出，可以直接使用alpha和1-alpha。 每一个分支都在做目标和背景的二分类
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)   # alpha_t *(1-y)^gamma
            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            # bce = p_t = y*p + (1-y)*(1-p)
            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            if torch.cuda.is_available():  # n-equal
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))



            # -------------------- 计算 【回归损失】 ------------------------------
            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]
                
                # 预测框
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]
                
                # 真实框
                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)
                
                # 将真实框 映射到 网络的输出空间
                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                # 通过anchor,将真实框的信息映射到RetinaNet网络的 输出空间上（targets/labels)，同网络在前向过程中的输出（predictions）一起求解损失函数
                negative_indices = 1 + (~positive_indices)  # （anchor_nums,1）
                # 只在正样本的anchor上计算
                regression_diff = torch.abs(targets - regression[positive_indices, :])
                ''' 
                x = target - prediction
                Smooth_L1_loss =    0.5 * x^2;  |x| <1
                                    |x| - 0.5;  otherwise 
                这里貌似是把系数改掉了
                '''                             
                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())
        return torch.stack(classification_losses).mean(dim=0, keepdim=True), 
                torch.stack(regression_losses).mean(dim=0, keepdim=True)

    
