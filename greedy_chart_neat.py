import matplotlib.pyplot as plt

# ================== 1. 原始的 Node 类设计 ==================
class Node:
    def __init__(self, name, is_start, is_end, heur_val):
        self.candidate = []      # 子节点列表
        self.is_start = is_start
        self.is_end = is_end
        self.name = name           # 名称
        self.heur_val = heur_val   # 启发函数值
        self.x = 0  # 节点的x坐标（用于绘图）
        self.y = 0  # 节点的y坐标（用于绘图）

    def set_candidate(self, aim):
        if aim not in self.candidate:
            self.candidate.append(aim)

# ================== 2. 创建节点 ==================
node_a = Node("a", True, False, 13)
node_b = Node("b", False, False, 10)
node_c = Node("c", False, False, 6)
node_d = Node("d", False, False, 12)
node_e = Node("e", False, False, 7)
node_f = Node("f", False, False, 8)
node_g = Node("g", False, False, 5)
node_h = Node("h", False, False, 3)
node_i = Node("i", False, False, 6)
node_j = Node("j", False, False, 3)
node_k = Node("k", False, True, 0)
node_l = Node("l", False, False, 6)

# ================== 3. 设置连接关系 ==================
node_a.set_candidate(node_b)
node_a.set_candidate(node_d)
node_a.set_candidate(node_e)
node_b.set_candidate(node_c)
node_c.set_candidate(node_b)
node_d.set_candidate(node_f)
node_d.set_candidate(node_a)
node_e.set_candidate(node_i)
node_e.set_candidate(node_h)
node_e.set_candidate(node_g)
node_e.set_candidate(node_a)
node_f.set_candidate(node_g)
node_f.set_candidate(node_d)
node_g.set_candidate(node_h)
node_g.set_candidate(node_k)
node_g.set_candidate(node_e)
node_g.set_candidate(node_f)
node_h.set_candidate(node_e)
node_h.set_candidate(node_g)
node_h.set_candidate(node_j)
node_i.set_candidate(node_e)
node_i.set_candidate(node_j)
node_j.set_candidate(node_i)
node_j.set_candidate(node_k)
node_j.set_candidate(node_h)
node_l.set_candidate(node_k)

# ================== 4. 设置节点位置（用于matplotlib绘图） ==================
positions = {
    node_a: (0, 0), node_b: (-1, 1), node_c: (-2, 1), node_d: (1, 0.5),
    node_e: (-1, -1), node_f: (2, 0.5), node_g: (0.5, -1.5),
    node_h: (-1.5, -2), node_i: (-2.5, -1), node_j: (-2.5, -2.5),
    node_k: (0, -2.5), node_l: (3, -2)
}

# 将坐标赋值给节点对象
for node, (x, y) in positions.items():
    node.x, node.y = x, y

# ================== 5. 贪婪搜索算法 ==================
def greedy_search(start_node):
    """贪婪最佳优先搜索"""
    visited = []  # 已访问节点
    path = []     # 路径
    current = start_node
    
    while not current.is_end:
        visited.append(current)
        path.append(current)
        
        # 获取未访问的子节点
        unvisited = [node for node in current.candidate if node not in visited]
        
        if not unvisited:
            print("搜索失败，无路可走")
            return path, visited
            
        # 选择启发值最小的节点
        current = min(unvisited, key=lambda x: x.heur_val)
    
    path.append(current)  # 加入目标节点
    visited.append(current)
    return path, visited

# 执行搜索
final_path, all_visited = greedy_search(node_a)
print(f"搜索路径: {' -> '.join([node.name for node in final_path])}")
print(f"访问顺序: {' -> '.join([node.name for node in all_visited])}")

# ================== 6. 使用matplotlib绘制图形 ==================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ---------- 子图1：完整图结构 ----------
ax1.set_title("完整图结构 (使用Node类)", fontsize=14, pad=20)

for node in positions.keys():
    # 根据节点类型设置颜色
    if node.is_start:
        color = '#ff9999'  # 起始节点-浅红
    elif node.is_end:
        color = '#66b3ff'  # 目标节点-浅蓝
    else:
        color = '#99ff99'  # 普通节点-浅绿
    
    # 绘制节点（圆形）
    circle = plt.Circle((node.x, node.y), 0.15, color=color, alpha=0.8)
    ax1.add_patch(circle)
    
    # 添加节点名称标签
    ax1.text(node.x, node.y, node.name, fontsize=12, fontweight='bold',
             ha='center', va='center')
    
    # 添加启发值标签
    ax1.text(node.x, node.y-0.2, f'h={node.heur_val}', fontsize=9,
             ha='center', va='top', color='darkred')


for node in positions.keys():
    for candidate in node.candidate:
        # 计算边的方向
        dx = candidate.x - node.x
        dy = candidate.y - node.y
        
        # 绘制箭头（有向边）
        ax1.arrow(node.x, node.y, dx*0.8, dy*0.8, 
                 head_width=0.08, head_length=0.1, 
                 fc='gray', ec='gray', alpha=0.6)

# 设置坐标轴
ax1.set_xlim(-3.5, 3.5)
ax1.set_ylim(-3.5, 2.5)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# ---------- 子图2：搜索过程 ----------
ax2.set_title("贪婪最佳优先搜索过程", fontsize=14, pad=20)

# 先绘制基础图
for node in positions.keys():
    circle = plt.Circle((node.x, node.y), 0.15, color='lightgray', alpha=0.4)
    ax2.add_patch(circle)
    ax2.text(node.x, node.y, node.name, fontsize=12, fontweight='bold',
             ha='center', va='center', alpha=0.4)
    
    # 绘制启发值
    ax2.text(node.x, node.y-0.2, f'h={node.heur_val}', fontsize=9,
             ha='center', va='top', color='gray', alpha=0.4)

# 绘制所有边
for node in positions.keys():
    for candidate in node.candidate:
        dx = candidate.x - node.x
        dy = candidate.y - node.y
        ax2.arrow(node.x, node.y, dx*0.8, dy*0.8, 
                 head_width=0.08, head_length=0.1, 
                 fc='lightgray', ec='lightgray', alpha=0.3)

# 逐步高亮显示搜索路径（模拟动画）
for i in range(len(final_path)):
    if i == 0:
        # 绘制起始节点
        start = final_path[0]
        circle = plt.Circle((start.x, start.y), 0.15, color='#ff9999', alpha=0.8)
        ax2.add_patch(circle)
        ax2.text(start.x, start.y, start.name, fontsize=12, 
                fontweight='bold', ha='center', va='center')
        continue
    
    prev = final_path[i-1]
    curr = final_path[i]
    
    # 绘制边（橙色）
    dx = curr.x - prev.x
    dy = curr.y - prev.y
    ax2.arrow(prev.x, prev.y, dx*0.8, dy*0.8, 
             head_width=0.08, head_length=0.1, 
             fc='orange', ec='orange', alpha=0.8, linewidth=2)
    
    # 绘制当前节点
    if curr.is_end:
        color = '#66b3ff'  # 目标节点
    else:
        color = '#ffcc00'  # 中间节点（黄色）
    
    circle = plt.Circle((curr.x, curr.y), 0.15, color=color, alpha=0.8)
    ax2.add_patch(circle)
    ax2.text(curr.x, curr.y, curr.name, fontsize=12, 
            fontweight='bold', ha='center', va='center')
    
    # 学习点：添加步骤说明
    step_text = f'步骤 {i}: 从 {prev.name} 到 {curr.name}'
    ax2.text(0.5, -0.05, step_text, transform=ax2.transAxes,
            ha='center', fontsize=12, 
            bbox=dict(facecolor='yellow', alpha=0.7))
    
    plt.pause(1.5)
    
    # 如果不是最后一步，清除步骤文本
    if i < len(final_path)-1:
        for txt in ax2.texts:
            if '步骤' in txt.get_text():
                txt.remove()

# 设置子图2坐标轴
ax2.set_xlim(-3.5, 3.5)
ax2.set_ylim(-3.5, 2.5)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#ff9999', label='起始节点 (a)'),
    Patch(facecolor='#66b3ff', label='目标节点 (k)'),
    Patch(facecolor='#ffcc00', label='搜索中节点'),
    Patch(facecolor='#99ff99', label='普通节点'),
    Patch(edgecolor='orange', facecolor='white', label='已探索路径', linewidth=3)
]
ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

# 学习点：调整布局并显示
plt.tight_layout()
print("\n正在显示可视化结果... 请查看弹出的窗口。")
plt.show()