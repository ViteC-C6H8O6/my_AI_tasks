import matplotlib.pyplot as plt

# ========================= 绘制图像 ===========================
plt.ion()
fig , ax = None , None
def plot_init():
    global fig , ax
    fig , ax = plt.subplots(figsize = (16,25))    #创建画布
    
    ax.set_xlim(0,50)
    ax.set_ylim(0,80)
    ax.set_aspect('equal')
    
    ax.set_title("一元线性回归演示")
    ax.set_xlabel("X坐标",fontsize=12)
    ax.set_ylabel("Y坐标",fontsize=12)

    return fig , ax 

fig , ax = plot_init()

def draw_picture(ax,x,y,n,a,b):
    # 画单点
    dir = (x,y)
    new_node = plt.Circle(dir,color='red',radius=0.5,alpha=0.8)
    ax.add_patch(new_node)
    # 画线
    # n = len(ax.patches)
    if n!=1 :
        nodex = (1,50)
        nodey = (1*a+b,59*a+b)
        ax.plot(nodex,nodey,color='gray',linewidth = 2.5,linestyle='--',alpha=0.7,label=f'第{n}次回归')
       
    ax.set_xlim(0,50)
    ax.set_ylim(0,80)
    ax.set_aspect('equal')
    fig.canvas.draw()
    fig.canvas.flush_events()


# ========================= 计算函数 ===========================
def linear_regression(n,x,y,sxy=None,sx2=None,sx=None,sy=None):
    # 单点处理
    if n==1 :
        sxyn = x*y
        sx2n = x**2
        sxn = x 
        syn = y
        a = None
        b = None
        return (a,b,sxyn,sx2n,sxn,syn) 
    
    # 多点画线
    else :
        sxn = sx + x
        syn = sy + y
        sxyn = sxy + x*y
        sx2n = sx2 + x**2

    a = (sxyn - sxn*syn/n)/(sx2n-sxn**2/n)
    b = (syn/n) - a*(sxn/n)

    return (a,b,sxyn,sx2n,sxn,syn)

# ========================= 加点函数 =========================
def additions(x,y,sxy,sx2,sx,sy,n):
    n+=1
    a,b,sxy,sx2,sx,sy = linear_regression(n,x,y,sxy,sx2,sx,sy)
    ax.lines[-1].set(color='gray',linewidth = 2.5,linestyle='--',alpha=0.7)
    draw_picture(ax,x,y,n,a,b)
    ax.lines[-1].set(color='red',linestyle='-',linewidth=3.5,alpha=1,zorder=100)
    print(f"result: a={a} b={b}")
    feedback = input("是否加点?是则输入点的坐标,如:3,2 否则输入#退出")
    if feedback=='#':
        fig.canvas.draw()
        fig.canvas.flush_events()
        return
    else:
        x_n,y_n = list(map(float,(feedback.strip().split(','))))
        additions(x_n,y_n,sxy,sx2,sx,sy,n)
        return


# ========================= 主程序 =========================    
def main():
    print("一元线性回归")
    print("="*50)
    x_set = list(map(float, input("请输入初始化x坐标集(例如:1,5,7...):").strip().split(',')))
    y_set = list(map(float, input("请输入初始化y坐标集(例如:1,5,7...):").strip().split(',')))

    # 初始化
    sxy = sx2 = sx = sy = 0
    n = 1

    #分点建立回归直线
    for (x,y) in list(zip(x_set,y_set)):
        a,b,sxy,sx2,sx,sy = linear_regression(n,(x,y)[0],(x,y)[1],sxy,sx2,sx,sy)
        draw_picture(ax,x,y,n,a,b)
        n+=1
    ax.lines[-1].set(color='red',linestyle='-',linewidth=3.5,alpha=1,zorder=100)
    plt.tight_layout()
    plt.show()    

    print(f"result: a={a} b={b}")
    feedback = input("是否加点?是则输入点的坐标,如:3,2 否则输入#退出")
    if feedback=='#':
        plt.ioff()
        plt.show(block = True)
        return
    else:
        x_n,y_n = list(map(float,(feedback.strip().split(','))))
        additions(x_n,y_n,sxy,sx2,sx,sy,n)
        return
        
    

if __name__ == "__main__":
    main()

