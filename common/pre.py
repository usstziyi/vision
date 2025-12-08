import torch
from .offset import offset_inverse
from .nms import nms

"""
    使用非极大值抑制来预测边界框
    假设我们在检测图像中的猫：
    - 图像中有一只猫(类别1)
    - 生成了1000个锚框
    - 只有10个锚框与猫有较好重叠(标记为类别1)
    - 其余990个锚框与猫无重叠(标记为类别0,即背景)
    - 在NMS后,一些低置信度的预测被标记为-1(忽略):这些锚框不是背景，是被丢弃的低质量前景
"""
# 用于测试阶段
def multibox_detection(cls_probs,       # cls_probs(B, NCLS, 锚框总数)：每个锚框的类别概率,包括背景0类别的概率
                       offset_preds,    # offset_preds(B, NAC*4)：每个锚框的偏移量预测
                       anchors,         # anchors(1, 锚框总数, 4):锚框生成器
                       nms_threshold=0.5,      # 非极大值抑制的IoU阈值，高于此阈值的重复框会被抑制
                       pos_threshold=0.009999999):  # 正类置信度阈值，低于此值的预测会被视为背景
    
    device = cls_probs.device

    batch_size = cls_probs.shape[0]
    num_classes = cls_probs.shape[1]
    num_anchors = cls_probs.shape[2] # 锚框总数

    # anchors(1, NAC, 4) -> anchors(NAC, 4)
    anchors = anchors.squeeze(0)



    out = []
    for i in range(batch_size):
        # cls_prob(NCLS, 锚框总数)
        cls_prob = cls_probs[i]
        # cls_prob[0]  通常表示背景类的概率
        # cls_prob[1:] 表示除了前景的所有类别概率
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
        # 把偏移量施加到锚框上，得到目标锚框(相对尺寸)
        target_anchors = offset_inverse(anchors, offset_pred)


        # 非极大值抑制：确保了最终输出结果没有冗余重叠框
        # target_anchors(锚框总数,4):模型预测的目标锚框(相对尺寸)
        # scores(锚框总数,):模型预测的目标锚框的最大类别概率
        # keep(R,):保留的anchor索引，R<=锚框总数
        keep = nms(target_anchors, scores, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        # combined(K+NAC,):保留的anchor索引+所有anchor索引
        combined = torch.cat((keep, all_idx))
        # 比如 [0, 2, 0, 1, 2, 3] 
        # uniques ：所有唯一元素 [0, 1, 2, 3] 
        # counts ：每个唯一元素在 combined 中出现的次数 [2, 1, 2, 1] 
        uniques, counts = combined.unique(return_counts=True)
        # 凡是counts为1的索引，都不在keep中，被认为是背景
        # non_keep(NAC-K,):所有不在keep中的anchor索引
        non_keep = uniques[counts == 1]
        # all_id_sorted(K+NAC-K,):保留的anchor索引+所有不在keep中的anchor索引
        # 重新排列(keep,non_keep),keep+non_keep=所有anchor索引
        all_id_sorted = torch.cat((keep, non_keep))
        # 不在keep中的anchor，类别索引下调到-1(背景类)
        class_id[non_keep] = -1

        # 按照排序(keep,non_keep)后的索引重新排列class_id数组
        # 按照排序(keep,non_keep)后的索引重新排列scores数组
        # 按照排序(keep,non_keep)后的索引重新排列预测框数组
        class_id = class_id[all_id_sorted]
        scores = scores[all_id_sorted] 
        predicted_bb = target_anchors[all_id_sorted] 
        

        # NMS可能保留了一些置信度不高但不与其他框重叠的预测框
        # 这些低置信度预测很可能是误检，需要进一步过滤
        # 再次筛选，将置信度低于阈值的预测框类别设为-1
        # pos_threshold是一个用于非背景预测的阈值
        # 当一个预测框被判定为背景时（因为置信度太低），我们用 1 - 原分数 来表示它作为背景的可能性
        # 再次更新
        below_min_idx = (scores < pos_threshold)
        class_id[below_min_idx] = -1
        scores[below_min_idx] = 1 - scores[below_min_idx]

        # class_id(NAC,1)
        # scores(NAC,1)
        # predicted_bb(NAC, 4)
        # pred_info(NAC, 6):(class_id, scores, predicted_bb)
        pred_info = torch.cat((class_id.unsqueeze(1), scores.unsqueeze(1), predicted_bb), dim=1)
        out.append(pred_info)
        # out(B, NAC, 6)
        # 调用 torch.stack(out) 时不指定 dim 参数，
        # 函数会沿着第 0 维（第一个维度）创建一个新的维度来堆叠输入张量序列
    return torch.stack(out)