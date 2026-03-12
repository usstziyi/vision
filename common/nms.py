import torch
from .iou import box_iou
from .anchor2label import offset_inverse
'''
    非极大值抑制非极大值抑制:删除置信度接近的锚框
    保留置信度最高的检测框，并抑制与其高度重叠（即交并比 IoU 较大）的其他检测框。
'''


# 非极大值抑制
# boxes: (N, 4)  N个边界框，每个框4个坐标
# scores: (N,)   每个边界框的置信度得分
# iou_threshold: 抑制阈值
# 混合类别的 NMS
def nms(boxes, scores, iou_threshold):
    # 对预测边界框的置信度进行排序
    # B: (N,)  按置信度降序排列的索引:从高到低
    sorted_score = torch.argsort(scores, dim=-1, descending=True)  
    keep_indices = []  # 保留预测边界框的指标

    # NMS
    while sorted_score.numel() > 0:# 当待处理框数量大于0时，继续处理
        # 本轮最高置信框索引
        max_indices = sorted_score[0]
        rest_indices = sorted_score[1:]
        # 保留当前最高置信框索引
        keep_indices.append(max_indices)
        # 如果只剩一个框，直接跳出循环
        if sorted_score.numel() == 1: 
            break


        
        # 取出当前最高置信框
        current_box = boxes[max_indices, :].reshape(-1, 4)          # (1, 4)
        # 取出剩余待比较框
        rest_boxes = boxes[rest_indices, :].reshape(-1, 4)       # (N-1, 4)
        # 计算两两 IoU(1,N-1)->(N-1,)
        # (0,1,2,3,4)
        # (0-1,0-2,0-3,0-4)
        iou = box_iou(current_box, rest_boxes).reshape(-1)  # (N-1,)
        # 筛选出 IoU 小于阈值的框自然索引
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)  # inds: (M,)  保留的索引，M <= N-1
        # 转换为总绝对索引
        inds = inds + 1
        # 下一轮参与NMS的框索引
        sorted_score = sorted_score[inds]

    return torch.tensor(keep_indices, device=boxes.device)  # (K,)  K为最终保留的框数量


# pytorch自带的nms函数
# from torchvision.ops import nms
# import torch

# # 假设 boxes 和 scores 已经定义
# # boxes: (N, 4) 格式为 [x1, y1, x2, y2]
# # scores: (N,)
# # iou_threshold: 标量

# keep_indices = nms(boxes, scores, iou_threshold)

# # keep_indices 是一个包含保留框索引的 Tensor (K,)
# # 你可以通过它来获取最终的框和得分：
# final_boxes = boxes[keep_indices]
# final_scores = scores[keep_indices]




"""
    使用非极大值抑制来预测质量的边界框
    假设我们在检测图像中的猫：
    - 图像中有一只猫(类别1)
    - 生成了1000个锚框
    - 只有10个锚框与猫有较好重叠(标记为类别1)
    - 其余990个锚框与猫无重叠(标记为类别0,即背景)
    - 在NMS后,一些低置信度的预测被标记为-1(忽略):这些锚框不是背景，是被丢弃的低质量前景
"""
# 用于测试阶段
def filter_boxes_by_nms(pred_classes,       # cls_probs(B, NCLS, 锚框总数)：每个锚框的类别概率,包括背景0类别的概率
                       offset_preds,    # offset_preds(B, NAC*4)：每个锚框的偏移量预测
                       anchors,         # anchors(1, 锚框总数, 4):锚框生成器
                       nms_threshold=0.5,      # 非极大值抑制的IoU阈值，高于此阈值的重复框会被抑制
                       pos_threshold=0.009999999):  # 正类置信度阈值，低于此值的预测会被视为背景
    
    device = pred_classes.device

    batch_size = pred_classes.shape[0]
    num_classes = pred_classes.shape[1]
    num_anchors = pred_classes.shape[2] # 锚框总数

    # anchors(1, NAC, 4) -> anchors(NAC, 4)
    anchors = anchors.squeeze(0)



    list_pred_info = []
    for i in range(batch_size):
        # cls_prob(NCLS, 锚框总数)
        cls_prob = pred_classes[i]
        # cls_prob[0]  通常表示背景类的概率
        # cls_prob[1:] 表示除了背景类的所有物体类别的概率
        # torch.max dim=0
        # 给每个anchor找到最大的类别概率和对应的类别索引
        # scores(NAC):每个anchor的最大类别概率
        # class_id(NAC):每个anchor的最大类别索引
        # 我们希望找到每个锚框最有可能属于哪个具体的物体类别（如猫、狗等）
        # 背景类别（0）只是用来标识"这里没有物体"，不需要参与物体类别的竞争
        # 后续处理的逻辑：
        # - 如果一个锚框的最大物体类别概率仍然很低（低于pos_threshold），则认为它是背景
        # - 这种设计允许模型明确区分三种状态：
        # - 明确检测到某个物体类别（高置信度）
        # - 确定是背景（低置信度）
        # - 不确定/忽略（通过NMS抑制或置信度过低设为-1）
        # 先判断是否有物体，再判断是什么物体
        # 这里的class_id0对应前景，不是背景
        # class_id 中的 0 对应 cls_prob[1] （第一个物体类别）
        # class_id 中的 1 对应 cls_prob[2] （第二个物体类别）
        # 举例：假设我们有3个物体类别：背景、狗、猫
        # 更符合实际情况的类别概率分布
        # cls_prob = torch.tensor([
        #     [0.9,  0.2,  0.1,  0.05],  # 背景类概率 (通常较高)
        #     [0.05, 0.7,  0.2,  0.03],  # 猫类概率
        #     [0.05, 0.1,  0.7,  0.92]   # 狗类概率
        # ])
        # cls_prob[1:] 切片后：
        # [
        #     [0.05, 0.7,  0.2,  0.03],  # 猫类概率
        #     [0.05, 0.1,  0.7,  0.92]   # 狗类概率
        # ]
        # torch.max 结果：
        # scores = [0.05, 0.7, 0.7, 0.92]  # 每个锚框的最大物体类别概率
        # class_id = [0, 0, 1, 1]           # 对应的类别索引（相对于切片后）
        # 实际类别映射：
        # 锚框0: 最可能是猫(类别1)，置信度0.05
        # 锚框1: 最可能是猫(类别1)，置信度0.7
        # 锚框2: 最可能是狗(类别2)，置信度0.7
        # 锚框3: 最可能是狗(类别2)，置信度0.92
        # 我们只关注前景类别的概率（索引1及以上）
        # scores(锚框总数):模型预测的目标锚框的类别概率(最大)
        # class_id(锚框总数):模型预测的目标锚框的类别索引
        # 背景类不应该与物体类别竞争
        # 后续处理 ：在后续代码中会有专门的逻辑处理背景情况：
        scores, class_id = torch.max(cls_prob[1:], dim=0)


        # offset_preds[i](锚框总数*4)
        # reshape(-1, 4):(锚框总数, 4)
        # offset_pred(锚框总数, 4)
        offset_pred = offset_preds[i].reshape(-1, 4)
        # 把偏移量施加到锚框上，得到预测框
        # anchors(锚框总数, 4)
        # offset_pred(锚框总数, 4)
        # target_anchors(锚框总数, 4):模型预测的目标锚框(相对尺寸)
        # 把偏移量施加到锚框上，得到预测框(相对尺寸)
        pred_boxes = offset_inverse(anchors, offset_pred)


        # 非极大值抑制：确保了最终输出结果没有冗余重叠框
        # target_anchors(锚框总数,4):模型预测的目标锚框(相对尺寸)
        # scores(锚框总数,):模型预测的目标锚框的最大类别概率
        # keep_indices(R,):保留的anchor索引，R<=锚框总数
        # 此处的scores里面是多种类型混在一起的分数排名
        # 混合类别的 NMS:先定类别，后做NMS,不会出现同一个预测框被分配为两个类别的情况
        # 如果两个重叠很高的框，一个是“猫（0.9）”，一个是“狗（0.8）”。
        # NMS 会保留分数高的“猫”，抑制掉“狗”。
        keep_indices = nms(pred_boxes, scores, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        mask = torch.zeros(num_anchors, dtype=torch.bool, device=device)
        mask[keep_indices] = True
        non_keep = torch.where(~mask)[0]


        # 重新排列(keep,non_keep),keep+non_keep=所有anchor索引
        all_id_sorted = torch.cat((keep_indices, non_keep))
        # 不在keep中的anchor，类别索引下调到-1(丢弃)
        class_id[non_keep] = -1



        # 按照排序(keep,non_keep)后的索引重新排列class_id数组
        # 按照排序(keep,non_keep)后的索引重新排列scores数组
        # 按照排序(keep,non_keep)后的索引重新排列预测框数组
        class_id = class_id[all_id_sorted]
        scores = scores[all_id_sorted] 
        pred_boxes = pred_boxes[all_id_sorted] 
        

        # non_keep中的class_id已经被设为-1(丢弃)
        # 还需要从keep中筛选出置信度低于阈值的预测框
        below_min_idx = (scores < pos_threshold)
        class_id[below_min_idx] = -1
        scores[below_min_idx] = 1 - scores[below_min_idx]

        # class_id(NAC,1)
        # scores(NAC,1)
        # predicted_bb(NAC, 4)
        # pred_info(NAC, 6):(class_id, scores, pred_boxes)
        pred_info = torch.cat((class_id.unsqueeze(1), scores.unsqueeze(1), pred_boxes), dim=1)
        list_pred_info.append(pred_info)
        # list_pred_info(B, NAC, 6)
        # 调用 torch.stack(out) 时不指定 dim 参数，
        # 函数会沿着第 0 维（第一个维度）创建一个新的维度来堆叠输入张量序列
    return torch.stack(list_pred_info, dim=0)