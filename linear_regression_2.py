#feature
X = [[10, 3], [20, 3], [25, 3], [28, 2.5], [30, 2], [35, 2.5], [40, 2.5]]
#lable
y = [60, 85, 100, 120, 140, 145, 163]
#参数
w = [0.0 , 0.0 , 0.0]
#学习率
lr = 0.0001
#迭代次数
num_iterate = 1000



def main() :
    
    '''for i in range(0, 7) :
        print("输入样本: pi ti")
        X.append(list(map(float , input().split())))

    for i in range(0, 7) :
        print("请输入标签: l1 l2 l3 ……")
        y = list(map(float , input().split()))'''

    for i in range(0 , num_iterate) : 

        #y的预测值
        y_pred = [w[0] + w[1] * x[0] + w[2] * x[1] for x in X]

        #误差
        loss = sum(((w[0] + w[1] * x[0] + w[2] * x[1]) ** 2 for x in X))

        #梯度分量
        grad_w0 = 2 * sum(y_pred[j] - y[j] for j in range(len(y))) / len(y)
        grad_w1 = 2 * sum((y_pred[j] - y[j]) * X[j][0] for j in range(len(y))) / len(y)
        grad_w2 = 2 * sum((y_pred[j] - y[j]) * X[j][1] for j in range(len(y))) / len(y)

        #更新
        w[0] = w[0] - lr * grad_w0
        w[1] = w[1] - lr * grad_w1
        w[2] = w[2] - lr * grad_w2

        print(f"loss: {loss}\n")
        '''if (loss < 1e9):
            break'''
    
    print(f"{w[0]} , {w[1]} , {w[2]}\n")



if __name__ == "__main__" :
    main()


