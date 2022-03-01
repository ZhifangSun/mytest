import math
import random

# 温度变化率
K = 0.97
# 结束温度，控制降温时间
EPS = 1E-10


# 计算自变量的增量
def get(T):
    return T * random.randint(-32767, 32767)


# 所计算的值与所有重物的势能成正比，要求总势能最小的点
def shinengcha(x, y, w, nowx, nowy):
    ans = 0
    for i in range(len(x)):
        ans += ((abs(nowx - x[i]) ** 2 + abs(nowy - y[i]) ** 2) ** 0.5) * w[i]
    return ans


def solve():
    n = eval(input())  #输入
    x, y, w = [0] * n, [0] * n, [0] * n
    for i in range(n):
        x[i], y[i], w[i] = map(eval, input().split())
    x0, y0 = sum(x) / n, sum(y) / n  #找到中间值
    ans = shinengcha(x, y, w, x0, y0)    #得到初试势能
    print(f'初试势能{ans}')
    cnt = 2
    while cnt > 0:
        cnt -= 1
        cur = ans
        tip=0
        x1, y1 = x0, y0
        T = 100000
        # 初始温度
        while T > EPS:
            x2, y2 = x1 + get(T), y1 + get(T)    #改变增量
            if tip<=3:
                print(f'增量后的坐标{x2} {y2}')
            temp = shinengcha(x, y, w, x2, y2)   #计算势能差
            if tip <= 3:
                print(f'势能差{temp}')
            if temp < ans:    #概率判断
                ans = temp
                x0, y0 = x2, y2
            if cur > temp or math.exp((cur - temp) / T) > random.random(): #热力学定理
                cur = temp
                x1, y1 = x2, y2
            T *= K  #退火
            tip+=1
    print("{:.3f} {:.3f}".format(x0, y0))


solve()

