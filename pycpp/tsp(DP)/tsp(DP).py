import operator
import numpy as np
import math
import matplotlib.pyplot as plt
from time import *
import tracemalloc
import copy
import random
import functools
#默认为升序
def cmp(a,b):
    if a<b:
        return -1
    else :
        return  1
#第一个元素升序，第二个元素降序
def mycmp(a,b):
    if a[0]==b[0]:
        return -cmp(a[1],b[1])
    else :
        return cmp(a[0],b[0])
CityNum = 200  # 城市数量
MinCoordinate = 0  # 二维坐标最小值
MaxCoordinate = 101  # 二维坐标最大值
#images_sorted = sorted(images,key=operator.attrgetter('storage'))
MyType=np.dtype([('a',int),('b',int)])
#coordinate = [(random.randint(MinCoordinate,MaxCoordinate),random.randint(MinCoordinate,MaxCoordinate)) for i in range(CityNum)]
coordinate=[(88, 16), (42, 76), (5, 76), (69, 13), (73, 56), (100, 100), (22, 92), (48, 74), (73, 46),(39, 1), (51, 75), (92, 2), (101, 44), (55, 26), (71, 27), (42, 81), (51, 91), (89, 54),(33, 18), (40, 78)]
#coordinate=[(1, 4), (2, 5), (3, 3), (4, 5), (5, 1), (6, 4)]
#coordinate=[[565.0,575.0],[25.0,185.0],[345.0,750.0],[945.0,685.0],[845.0,655.0],[880.0,660.0],[25.0,230.0],[525.0,1000.0],[580.0,1175.0],[650.0,1130.0],[1605.0,620.0],[1220.0,580.0],[1465.0,200.0],[1530.0,  5.0],[845.0,680.0],[725.0,370.0],[145.0,665.0],[415.0,635.0],[510.0,875.0],[560.0,365.0],[300.0,465.0],[520.0,585.0],[480.0,415.0],[835.0,625.0],[975.0,580.0],[1215.0,245.0],[1320.0,315.0],[1250.0,400.0],[660.0,180.0],[410.0,250.0],[420.0,555.0],[575.0,665.0],[1150.0,1160.0],[700.0,580.0],[685.0,595.0],[685.0,610.0],[770.0,610.0],[795.0,645.0],[720.0,635.0],[760.0,650.0],[475.0,960.0],[95.0,260.0],[875.0,920.0],[700.0,500.0],[555.0,815.0],[830.0,485.0],[1170.0, 65.0],[830.0,610.0],[605.0,625.0],[595.0,360.0],[1340.0,725.0],[1740.0,245.0]]
#coordinate=[(1, 5), (3, 4), (4, 0), (5, 3), (5, 6), (8, 5), (6, 1), (7, 3)]
def f(a,b):
    return math.sqrt((coordinate[a][0]-coordinate[b][0])**2*1.0+(coordinate[a][1]-coordinate[b][1])**2*1.0)
def draw_path(line, CityCoordinates):
    '''
    #画路径图    然而用我说的那个思路的话走的路线应该是这样的
    输入：line-路径，CityCoordinates-城市坐标；
    输出：路径图
    '''
    x, y = [], []
    for i in line:
        Coordinate = CityCoordinates[i]
        x.append(Coordinate[0])
        y.append(Coordinate[1])
    x.append(x[0])
    y.append(y[0])
    plt.plot(x, y, 'r-', color='#4169E1', alpha=0.8, linewidth=0.8)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
"""
n=int(input())
for i in range(n):
    q=list(map(int, input().split()))
    coordinate.append((q[0],q[1]))
print(f'输入的坐标{coordinate}')
"""
begin_time = time()
tracemalloc.start()
coordinate.sort(key=functools.cmp_to_key(mycmp))
#coordinate=sorted(coordinate,key=lambda x:x[0])
print(f'按照横坐标排序后的坐标{coordinate}')
print(len(coordinate))
dp=[None]*len(coordinate)  #dp[i][j]用来保存从起点出发的两条终点分别为i，j路径长度和的最小值
weight=[None]*len(coordinate)  #weight[i][j]用来记录i,j之间的距离
for i in range(len(coordinate)):   #初始化数组
    dp[i]=[0]*len(coordinate)
    weight[i] = [0] * len(coordinate)
#print(f'初始化dp{dp}')
#print(f'初始化weight{weight}')
for i in range(len(coordinate)):    #计算两点间距离
    for j in range(len(coordinate)):
        dp[i][j]=-1.0
        if i==j:
            weight[i][j]=0.0
        else:
            weight[i][j]=f(i,j)
#print(f'输出计算后的距离{weight}')
dp[0][0]=0
dp[1][0]=weight[1][0]
x,y=[],[]
a,path=[],[]
for i in range(len(coordinate)):
    path.append([])
    for j in range(len(coordinate)):
        path[i].append([])
path[1][0].append(0)
path[1][0].append(1)
for i in range(len(coordinate)):
    x.append(coordinate[i][0])
    y.append(coordinate[i][1])
#print(x)
#print(y)
plt.plot(x,y,'ro')
plt.show()
for i in range(2,len(coordinate)):
    for j in range(i):  #降低复杂度
        if i!=j+1:
            #plt.plot(coordinate[j])
            #plt.show()
            dp[i][j]=dp[i-1][j]+weight[i][i-1]
            path[i][j]=copy.deepcopy(path[i-1][j])
            if path[i][j][0]<path[i][j][len(path[i][j])-1]:
                path[i][j].append(copy.deepcopy(i))
            else:
                path[i][j].insert(0, i)
            print(path[i][j])
            #if j<=3:
                #print(dp)
        else:
            minn=float("inf")
            temp=-1
            for k in range(j):
                if dp[j][k]+weight[i][k]<minn:
                    minn=dp[j][k]+weight[i][k]
                    temp=k
                #minn=min(minn,dp[j][k]+weight[i][k])  #按理应该求解dp[k][j]+weight[i][k]
            dp[i][j]=minn
            path[i][j]=copy.deepcopy(path[j][temp])#但是我们知道dp[k][j]=dp[j][k]，并且我们只求了
            if path[i][j][0] < path[i][j][len(path[i][j]) - 1]:
                path[i][j].insert(0,i)
            else:
                path[i][j].append(copy.deepcopy(i))
            print(path[i][j])
ans=float("inf")
Line=path[len(coordinate)-1][len(coordinate)-2]
print(path[len(coordinate)-1][len(coordinate)-2])#表的上半部分，所以可以直接用dp[j][k]计算
for i in range(len(coordinate)-1):
    ans=min(ans,dp[len(coordinate)-1][i]+weight[len(coordinate)-1][i])
print(f'最终结果{dp}')
print(ans)
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
tracemalloc.stop()
end_time = time()
run_time = end_time - begin_time
print(run_time)
draw_path(Line,coordinate)