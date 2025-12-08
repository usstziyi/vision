import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

'''
    显示某个像素为中心的所有边界框
'''
# anchors(NAC,4):(xmin, ymin, xmax, ymax)
def show_anchors(axes, anchors, labels=None, colors=None):

    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    
    # anchors(NAC,4):(xmin, ymin, xmax, ymax)
    for i, bbox in enumerate(anchors):
        color = colors[i % len(colors)] # 循环使用颜色列表
        rect = d2l.bbox_to_rect(bbox.detach().cpu().numpy(), color)
        axes.add_patch(rect)
        # 每个边界框上添加标签文本
        if labels and len(labels) > i:
            # 如果边界框颜色是白色( 'w' )，则文本颜色设为黑色( 'k' )，否则文本颜色设为白色( 'w' )
            text_color = 'k' if color == 'w' else 'w'
            # bbox=dict(facecolor=color, lw=0) ：文本背景框设置，使用与边界框相同的颜色
            axes.text(rect.xy[0], rect.xy[1],  # xmin, ymin
                      labels[i],
                      # 文本会在指定坐标点精确居中显示
                      va='center', ha='center', 
                      fontsize=9, color=text_color,
                      # facecolor=color ：设置背景框的填充颜色，使用与边界框相同的颜色
                      # lw=0 ：设置背景框边框的线宽为0，即不显示边框
                      bbox=dict(facecolor=color, lw=0))


def display(img, output, threshold):
    d2l.set_figsize((5, 5))
    fig = plt.imshow(img)
    
    anchors = []
    labels = [] 
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        anchor = row[2:6] * torch.tensor((w, h, w, h), device=row.device)
        anchors.append(anchor)
        #'%.2f' % score 是Python中的字符串格式化语法，将浮点数格式化为保留两位小数的字符串
        labels.append('%.2f' % score)
    show_anchors(fig.axes, anchors, labels)
    plt.show()  
