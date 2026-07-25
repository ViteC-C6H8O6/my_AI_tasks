import matplotlib.pyplot as plt

# ================== 1. 原始的 Node 类设计 ==================
class Node:
    def __init__(self, val, name=None, a_or_b=None, children=None):
        self.val = val
        self.name = name
        self.children = children if children is not None else []
        self.type = a_or_b  # a是max，b是min
        self.x = 0
        self.y = 0
    
    def is_terminal(self):
        return self.val is not None and not self.children

# ================== 2. 构建决策树 ==================
leaf1 = Node(3, "叶子1")
leaf2 = Node(5, "叶子2")
leaf3 = Node(6, "叶子3")
leaf4 = Node(2, "叶子4")
leaf5 = Node(8, "叶子5")
leaf6 = Node(1, "叶子6")
leaf7 = Node(9, "叶子7")
leaf8 = Node(0, "叶子8")

node_d = Node(None, "D(max)", 'a', [leaf1, leaf2])
node_e = Node(None, "E(max)", 'a', [leaf3, leaf4])
node_f = Node(None, "F(max)", 'a', [leaf5, leaf6])
node_g = Node(None, "G(max)", 'a', [leaf7, leaf8])

node_b = Node(None, "B(min)", 'b', [node_d, node_e])
node_c = Node(None, "C(min)", 'b', [node_f, node_g])
node_a = Node(None, "A(max)", 'a', [node_b, node_c])

# ================== 3. 设置节点坐标（树状布局） ==================
# 手动设置树状布局坐标
positions = {
    # 第1层（根节点）
    node_a: (0, 3),
    
    # 第2层
    node_b: (-3, 2),
    node_c: (3, 2),
    
    # 第3层
    node_d: (-4.5, 1),
    node_e: (-1.5, 1),
    node_f: (1.5, 1),
    node_g: (4.5, 1),
    
    # 第4层（叶子节点）
    leaf1: (-5.5, 0),
    leaf2: (-3.5, 0),
    leaf3: (-2, 0),
    leaf4: (-1, 0),
    leaf5: (1, 0),
    leaf6: (2, 0),
    leaf7: (4, 0),
    leaf8: (5, 0)
}

for node, (x, y) in positions.items():
    node.x, node.y = x, y

# ================== 4. 查找函数 ==================
def find_by_dfs(aim, start):
    """深度优先查找节点"""
    if aim == start.name:
        return start
    for child in start.children:
        found = find_by_dfs(aim, child)
        if found:
            return found
    return None

# ================== 5. 绘图函数 ==================
def draw_tree(ax, highlight_node=None, visited_nodes=None, pruned_nodes=None):
    """绘制决策树"""
    if visited_nodes is None:
        visited_nodes = []
    if pruned_nodes is None:
        pruned_nodes = []
    
    # 绘制所有边
    all_nodes = [node_a, node_b, node_c, node_d, node_e, node_f, node_g, 
                 leaf1, leaf2, leaf3, leaf4, leaf5, leaf6, leaf7, leaf8]
    
    for node in all_nodes:
        for child in node.children:
            # 检查边是否被剪枝
            edge_color = 'gray'
            linewidth = 1
            linestyle = '-'
            
            # 如果子节点在剪枝列表中，用虚线表示
            if child in pruned_nodes or node in pruned_nodes:
                linestyle = '--'
                edge_color = 'red'
                linewidth = 2
            
            ax.plot([node.x, child.x], [node.y, child.y], 
                   color=edge_color, linewidth=linewidth, linestyle=linestyle, alpha=0.6)
    
    # 绘制所有节点
    for node in all_nodes:
        # 确定节点颜色和大小
        if node == highlight_node:
            color = 'gold'
            size = 0.4
            edgecolor = 'red'
            linewidth = 3
        elif node in visited_nodes:
            color = 'lightgreen'
            size = 0.3
            edgecolor = 'green'
            linewidth = 2
        elif node.type == 'a':  # MAX节点
            color = '#FF9999'  # 浅红
            size = 0.25
            edgecolor = 'darkred'
            linewidth = 1
        elif node.type == 'b':  # MIN节点
            color = '#66B3FF'  # 浅蓝
            size = 0.25
            edgecolor = 'darkblue'
            linewidth = 1
        else:  # 叶子节点
            color = '#99FF99'  # 浅绿
            size = 0.2
            edgecolor = 'darkgreen'
            linewidth = 1
        
        # 绘制节点（圆形）
        circle = plt.Circle((node.x, node.y), size, 
                           facecolor=color, edgecolor=edgecolor, 
                           linewidth=linewidth, alpha=0.8)
        ax.add_patch(circle)
        
        # 添加节点标签
        label = node.name
        if node.is_terminal():
            label += f'\n{node.val}'  # 叶子节点显示值
        
        ax.text(node.x, node.y, label, 
               ha='center', va='center', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # 设置坐标轴
    ax.set_xlim(-7, 7)
    ax.set_ylim(-1, 4)
    ax.set_aspect('equal')
    ax.axis('off')  # 关闭坐标轴
    
    # 添加图例
    ax.text(-6.5, 3.8, 'MAX节点', color='darkred', fontsize=10)
    ax.text(-6.5, 3.5, 'MIN节点', color='darkblue', fontsize=10)
    ax.text(-6.5, 3.2, '叶子节点', color='darkgreen', fontsize=10)
    ax.text(-6.5, 2.9, '当前节点', color='red', fontsize=10)
    ax.text(-6.5, 2.6, '已访问', color='green', fontsize=10)
    ax.text(-6.5, 2.3, '剪枝边', color='red', fontsize=10)

# ================== 6. Alpha-Beta剪枝算法（带可视化） ==================
def alpha_beta_with_viz(root, stander, visited_nodes=None, pruned_nodes=None, ax=None):
    """带可视化的Alpha-Beta剪枝"""
    if visited_nodes is None:
        visited_nodes = []
    if pruned_nodes is None:
        pruned_nodes = []
    
    # 记录当前节点
    visited_nodes.append(root)
    
    # 可视化：清除画布并重新绘制
    if ax:
        ax.clear()
        draw_tree(ax, highlight_node=root, visited_nodes=visited_nodes, pruned_nodes=pruned_nodes)
        ax.set_title(f"当前节点: {root.name}", fontsize=14, fontweight='bold')
        plt.pause(1)  # 暂停1秒，观察
    
    # 如果是叶子节点
    if root.is_terminal():
        value = root.val
        print(f"找到叶子节点{root.name}, 返回值: {value}")
        return value
    
    print(f"当前位置: {root.name}")
    
    if root.type == 'a':  # MAX节点
        print("这是一个最大节点")
        value = -float('inf')
        
        for i, child in enumerate(root.children):
            # 递归搜索子节点
            pass_value = alpha_beta_with_viz(child, value, visited_nodes, pruned_nodes, ax)
            value = max(value, pass_value)
            
            # Alpha剪枝
            if value >= stander:
                print(f"*发生了Alpha剪枝: {child.name}")
                # 记录剩余未访问的兄弟节点为剪枝节点
                for pruned_child in root.children[i+1:]:
                    pruned_nodes.append(pruned_child)
                break
        
        print(f"{root.name}返回值 {value}")
        return value
    
    else:  # MIN节点
        print("这是一个最小节点")
        value = float('inf')
        
        for i, child in enumerate(root.children):
            # 递归搜索子节点
            pass_value = alpha_beta_with_viz(child, value, visited_nodes, pruned_nodes, ax)
            value = min(value, pass_value)
            
            # Beta剪枝
            if value <= stander:
                print(f"*发生了Beta剪枝: {child.name}")
                # 记录剩余未访问的兄弟节点为剪枝节点
                for pruned_child in root.children[i+1:]:
                    pruned_nodes.append(pruned_child)
                break
        
        print(f"{root.name}返回值 {value}")
        return value

# ================== 7. 主程序 ==================
if __name__ == "__main__":
    # 创建图形窗口
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 绘制完整的决策树（左侧）
    draw_tree(ax1)
    ax1.set_title("完整的Alpha-Beta剪枝决策树", fontsize=16, fontweight='bold')
    
    # 右侧用于动态显示搜索过程
    ax2.set_title("Alpha-Beta剪枝搜索过程", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.ion()  # 开启交互模式
    plt.show()
    
    # 用户输入
    root_name = input("请输入要搜索的根节点名称（例如: A(max)）: ")
    
    # 查找节点
    start_node = find_by_dfs(root_name, node_a)
    
    if start_node:
        print("\n" + "="*50)
        print("开始Alpha-Beta剪枝搜索:")
        print("="*50)
        print("决策树结构:")
        print("        A(MAX)")
        print("       /      \\")
        print("    B(MIN)    C(MIN)")
        print("    /   \\    /   \\")
        print("   D     E  F     G")
        print("  / \\   / \\/ \\   / \\")
        print("  3 5   6 2 8 1   9 0")
        print("="*50)
        
        # 执行带可视化的Alpha-Beta剪枝
        visited_nodes = []
        pruned_nodes = []
        result = alpha_beta_with_viz(start_node, float('inf'), visited_nodes, pruned_nodes, ax2)
        
        print("="*50)
        print(f"最终结果: {result}")
        print(f"访问过的节点: {[node.name for node in visited_nodes]}")
        print(f"被剪枝的节点: {[node.name for node in pruned_nodes]}")
        
    else:
        print(f"未找到节点: {root_name}")
    
    # 保持窗口显示
    plt.ioff()  # 关闭交互模式
    plt.show()