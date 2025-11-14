import torch
from .offset import offset_inverse
from .nms import nms

#
def multibox_detection(cls_probs,       # cls_probs(1, NCLS, NAC)
                       offset_preds,    # offset_preds(1, NAC*4)
                       anchors,         # anchors(1, NAC, 4)
                       nms_threshold=0.5,      # 非极大值抑制的IoU阈值，高于此阈值的重复框会被抑制
                       pos_threshold=0.009999999):  # 正类置信度阈值，低于此值的预测会被视为背景
    """使用非极大值抑制来预测边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    # anchors(1, NAC, 4) -> anchors(NAC, 4)
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]

    out = []
    for i in range(batch_size):
        # cls_prob(NCLS, NAC)
        # offset_preds[i](NAC*4)
        # reshape(-1, 4):(NAC, 4)
        # offset_pred(NAC, 4)
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        # cls_prob[0] 通常表示背景类的概率
        # cls_prob[1:] 表示除了背景类之外的所有类别概率
        # torch.max 函数用于找出张量中的最大值:列
        # conf(NAC)
        # class_id(NAC)
        conf, class_id = torch.max(cls_prob[1:], dim=0)
        # 把偏移量施加到锚框上，得到预测框
        # anchors(NAC, 4)
        # offset_pred(NAC, 4)
        # predicted_bb(NAC, 4)
        predicted_bb = offset_inverse(anchors, offset_pred)
        # 非极大值抑制
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        # 比如 [0, 2, 0, 1, 2, 3] 
        # uniques ：所有唯一元素 [0, 1, 2, 3] 
        # counts ：每个唯一元素在 combined 中出现的次数 [2, 1, 2, 1] 
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        
        conf = conf[all_id_sorted]
        predicted_bb = predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]

        # class_id(NAC,1)
        # conf(NAC,1)
        # predicted_bb(NAC, 4)
        # pred_info(NAC, 6)
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
        # out(1, NAC, 6)
    return torch.stack(out)