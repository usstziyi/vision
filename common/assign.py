import torch
from .iou import box_iou

'''
    正负样本分配:其目的是决定哪些锚框负责预测哪个目标，并用于后续的损失计算和训练。
    绑定gt到ac,确保每个gt都被分配到anchor
    NAC个anchor中:
    1. 正样本anchor都有一个对应的gt索引:0,1,...
    2. 负样本anchor没有对应的gt索引:-1

    我们需要回答两个问题：
    哪些锚框是“正样本”（负责预测某个 GT）？
    哪些锚框是“负样本”（背景，不包含目标）？

    第一阶段4：从每行中选出最大的IoU对应的真实边界框(列)，绑定当当前锚框(行)
    第二阶段123：从jaccard矩阵中选出最大IoU值的索引，做行列绑定，并删除行列

    贪心匹配 + 可选扩展匹配
    开始
    │
    ├─ 构建 IoU 矩阵 X (n_a × n_b)
    │
    ├─ Step 1: 找全局最大 IoU → 分配一对 (A_i, B_j)，删该行该列
    │
    ├─ Step 2: 在剩余矩阵中重复 Step 1，直到所有真实框都被分配
    │
    ├─ Step 3: 此时已有 n_b 个锚框被分配（每个真实框一个）
    │
    └─ Step 4: 对剩下 n_a - n_b 个锚框，查原始矩阵中"每行"最大 IoU，若 > 阈值，则额外分配对应的真实框
'''
# 将最接近的真实边界框分配给锚框
# ground_truth(NGT,4):(xmin,ymin,xmax,ymax)
# anchors(NAC,4):(xmin,ymin,xmax,ymax)
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框
    参数:
        ground_truth: 真实边界框，形状为 (NGT, 4)
        anchors: 锚框，形状为 (NAC, 4)
        device: 计算设备（如 'cpu' 或 'cuda'）
        iou_threshold: IoU 阈值，用于判断是否分配真实边界框，默认为 0.5
    返回:
        anchors_bbox_map: 每个锚框分配到的真实边界框索引，形状为 (NAC,)
    """

    # 候选锚框数量
    num_gt_boxes = ground_truth.shape[0]
    # 真实锚框数量
    num_anchors = anchors.shape[0]

    # 计算交并比
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    # jaccard(num_anchors,num_gt_boxes)
    jaccard = box_iou(anchors, ground_truth)
    # 创建一个全为-1的张量，用于存储每个锚框分配到的真实边界框索引
    # anchors_bbox_map(NAC,):-1 表示未分配, 0 表示背景
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,device=device)

    # 找每行最大IoU
    # 每行最大值:max_ious(num_anchors,)
    # 每行最大值的列索引:indices(num_anchors,)：对应的真实边索引
    max_ious, indices = torch.max(jaccard, dim=1)

    # 从Max IoU中筛选出正样本Positive IoU
    # max_ious >= iou_threshold:从选出来的最大值里筛选出IoU大于阈值的锚框索引，作为正样本
    # torch.nonzero():返回非零元素(正样本)的索引,返回值是二维张量，所以要reshape(-1)转换为一维张量
    # num_positive:此时正样本数量:max_ious >= iou_threshold的元素数量
    # anc_i(num_positive,):IoU矩阵的行索引，对应锚框索引
    anc_idx = torch.nonzero(max_ious >= iou_threshold).reshape(-1) # 哪些行此时是正样本
    # box_idx(num_positive,):IoU矩阵的列索引，对应真实边界框索引
    box_idx = indices[max_ious >= iou_threshold]

    # 先执行第4步：从每行中选出最大的IoU对应的真实边界框(列)，绑定当当前锚框(行)
    anchors_bbox_map[anc_idx] = box_idx



    # ------------------------------------------------------- #

    # 执行第123步：从jaccard矩阵中选出最大IoU值的索引，做行列绑定，并删除行列
    # 轮询列轮，把所有真实边界框都分配给锚框
    for _ in range(num_gt_boxes):
        # 每轮从IoU矩阵(num_anchors,num_gt_boxes)中找到最大IoU值的索引
        # torch.argmax 默认将输入张量展平成一维，再返回最大值的“扁平”索引
        # 例如 jaccard 形状为 (num_anchors, num_gt_boxes)，则 max_idx 是 0 到 num_anchors*num_gt_boxes-1 的整数
        max_idx = torch.argmax(jaccard) 

        # 定位最大IoU值对应的锚框索引(行索引)
        anc_idx = (max_idx / num_gt_boxes).long()
        # 定位最大IoU值对应的真实边界索引(列索引)
        box_idx = (max_idx % num_gt_boxes).long()

        # 第二次映射：确保每个真实边界框至少有一个锚框负责预测（保障召回）
        # 即使某个真实边界框和所有锚框的 IoU 都 < 0.5，也要强制将它分配给 IoU 最大的那个锚框。
        # 这是为了防止某些小目标或特殊形状的目标完全“漏掉”，没有锚框负责预测它，从而导致训练时无法学习到这些目标。
        # 第二次映射强制一对一绑定，即使 IoU 较低，也要保证每个 GT 有“专属 anchor”用于回归学习。
        # 因此，第二次映射覆盖第一次的结果，是合理的策略调整，目的是提升召回率（recall）。
        # 建立锚框索引(行索引)与真实边界框索引(列索引)的映射关系
        anchors_bbox_map[anc_idx] = box_idx

        # discard:将最大IoU值对应的锚框索引(行索引)的IoU值设为-1，
        jaccard[:, box_idx] = torch.full((num_anchors,), -1)
        # discard:将最大IoU值对应的真实边界框索引(列索引)的IoU值设为-1，
        # 表示该真实边界框已经被分配到了锚框，下一轮不再参与找最大值
        jaccard[anc_idx, :] = torch.full((num_gt_boxes,), -1)
    # 输出(NAC,)
    return anchors_bbox_map