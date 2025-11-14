import torch
from .iou import iou

'''
    绑定gt到ac,确保每个gt都被分配到anchor
'''

# ground_truth(NGT,4):(xmin,ymin,xmax,ymax)
# anchors(NAC,4):(xmin,ymin,xmax,ymax)
# iou_threshold: IoU 阈值，用于判断是否分配真实边界框，默认为 0.5
# anchors_gt_map: 每个锚框分配到的真实边界框索引，形状为 (NAC,)
def bind_ground_truth_to_anchor(ground_truth, anchors, device, iou_threshold=0.5):

    # 候选锚框数量
    num_gt = ground_truth.shape[0]
    # 真实锚框数量
    num_anchor = anchors.shape[0]

    # 计算交并比
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    # anchors_iou:(NAC,NGT)
    anchors_iou = iou(anchors, ground_truth)    

    # 创建一个全为-1的张量，用于存储每个锚框分配到的真实边界框索引
    # anchors_gt_map(NAC,):-1表示anchor未bind到gt,anchor是负样本
    anchors_gt_map = torch.full((num_anchor,), -1, dtype=torch.long, device=device)

    # 找每行最大IoU
    # max_ious(NAC,):每行最大IoU
    # gt_ids(NAC,):每行最大IoU所在的列索引:对应的GT索引,范围：[0,NGT-1]
    max_ious, gt_ids = torch.max(anchors_iou, dim=1)

    # 从Max IoU中筛选出正样本Positive IoU
    # torch.nonzero():返回非零元素(正样本)的索引
    # torch.nonzero返回的shape是(P,1)，需要reshape(-1)
    # positive_ac_ids(P,):[0,NAC-1]
    positive_ac_ids = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    # 把gt_ids中对应正样本的索引提取出来,舍弃负样本的索引
    # positive_gt_ids(P,):IoU矩阵的列索引，对应GT索引  
    positive_gt_ids = gt_ids[max_ious >= iou_threshold]

    # 第一次映射：基于 IoU 阈值的正样本分配（常规匹配）
    # 第一次映射允许多对一（多个 anchor → 同一个 GT）
    # 但是由于负样本的存在，anchors_gt_map中可能有没有被分配的锚框索引，保留-1
    # 建立正样本索引(行索引)与真实边界框索引(列索引)的映射关系
    anchors_gt_map[positive_ac_ids] = positive_gt_ids

    # col_discard:纵向黑板擦
    # row_discard:横向黑板擦
    col_discard = torch.full((num_anchor,), -1)
    row_discard = torch.full((num_gt,), -1)

    # 一共要进行num_gt轮，确保每个gt都被分配到anchor
    for _ in range(num_gt):
        # 每轮从IoU矩阵(num_anchor,num_gt)中找到最大IoU值的索引
        # torch.argmax 默认将输入张量展平成一维，再返回最大值的“扁平”索引
        # 例如 anchors_iou 形状为 (num_anchor, num_gt)，则 max_iou_id 是 0 到 num_anchor*num_gt-1 的整数
        max_iou_id = torch.argmax(anchors_iou) 

        # 定位最大IoU值对应的锚框索引(行索引)
        ac_id = (max_iou_id / num_gt).long()
        # 定位最大IoU值对应的真实边界索引(列索引)
        gt_id = (max_iou_id % num_gt).long()


        # 第二次映射：确保每个gt都被分配到anchor
        # 第二次绑定的优先级更高:如果最优解在前，剩下的只能绑定低质量GT,指定在之后的判断中被舍弃掉
        # 即使某个真实边界框和所有锚框的 IoU 都 < 0.5，也要强制将它分配给 IoU 最大的那个锚框。
        # 这是为了防止某些小目标或特殊形状的目标完全“漏掉”，没有锚框负责预测它，从而导致训练时无法学习到这些目标。
        # 因此，第二次映射覆盖第一次的结果，是合理的策略调整，目的是提升召回率（recall）。
        # 建立锚框索引(行索引)与真实边界框索引(列索引)的映射关系
        anchors_gt_map[ac_id] = gt_id

        # 执行纵向擦除
        anchors_iou[:, gt_id] = col_discard
        # 执行横向擦除
        anchors_iou[ac_id, :] = row_discard
    # 经过两轮的bind,anchors_gt_map中每个gt都被分配到了anchor
    # 但是还有的anchor没有被分配到gt,这些anchor的索引在anchors_gt_map中为-1
    # 输出(NAC,)
    return anchors_gt_map
