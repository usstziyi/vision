import torch
import matplotlib.pyplot as plt




# 显示边界框bbox/锚框anchor
# boxes:盒子的实际坐标，格式为(左上x,左上y,右下x,右下y)
def show_boxes(axes, boxes, labels=None,linewidth=2, colors=None):
    """显示某个像素为中心的所有边界框"""
    # 安全检查
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = _make_list(obj=labels, default_values=[])
    colors = _make_list(obj=colors, default_values=['b', 'g', 'r', 'm', 'c'])
    
    # 遍历所有边界框
    for i, box in enumerate(boxes):
        color = colors[i % len(colors)] # 循环使用颜色列表
        # 盒子
        rect = plt.Rectangle(xy=(box[0], box[1]), width=box[2]-box[0], height=box[3]-box[1],fill=False, edgecolor=color, linewidth=linewidth)
        axes.add_patch(rect) # 把矩形数据添加到绘图区(axes)中
        # 盒子标签
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w' # 文本颜色，与边界框颜色相反
            bounding_box  = {'facecolor': color, 'lw': 0} # 文本背景框设置，使用与边界框相同的颜色
           
            axes.text(rect.xy[0],          # 文本左下角在图中的横坐标（锚点为左下角）
                      rect.xy[1],          # 文本左下角在图中的纵坐标
                      labels[i],           # 要显示的文本内容（当前边界框对应的标签）
                      va='center',         # 垂直对齐方式：居中
                      ha='center',         # 水平对齐方式：居中
                      fontsize=9,          # 字体大小
                      color=text_color,    # 字体颜色（与边界框颜色形成对比）
                      bbox=bounding_box)   # 文本背景框样式字典
                      
