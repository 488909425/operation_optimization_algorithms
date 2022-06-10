import numpy as np
import scipy as sp
import scipy.spatial as ssp
import math
import argparse
import elkai   # 专门用于解决TSP的库，但不适合于浮点数运算
import matplotlib.pyplot as plt


"命令令和参数解析"
def get_config(args=None):
    #使用 argparse 的第一步是创建一个 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Meta optimization")
    #help：对参数的简单，描述
    parser.add_argument('--customers_num', type=int, default=20, help="客户最大数量")
    parser.add_argument('--capacity', type=float, default=80, help="车辆最大容量")
    parser.add_argument('--EPSILON', type=float, default=0.00000000001, help="一个非常小的数值")
    parser.add_argument('--map_x_max', type=float, default=100, help="x轴最大范围")
    parser.add_argument('--map_y_max', type=float, default=100, help="y轴最大范围")
    parser.add_argument('--map_x_min', type=float, default=0, help="x轴最小范围")
    parser.add_argument('--map_y_min', type=float, default=0, help="y轴最小范围") 
    parser.add_argument('--max_demand', type=float, default=20, help="需求上界")
    parser.add_argument('--min_demand', type=float, default=5, help="需求下界")    
    parser.add_argument('--destroy_size', type=int, default=3, help="破坏算子数量")
    parser.add_argument('--repair_size', type=int, default=2, help="修复算子数量")
    parser.add_argument('--ρ', type=float, default=0.7, help="权值更新参数")
    config = parser.parse_args(args=[])   # 在交互环境下设置args=[]，在pycharm等编辑器则不用设置为[]
    return config

#参数实例化
config = get_config()   

"计算距离矩阵，分别传入x,y的坐标array，返回距离矩阵"
def genDistanceMat(x,y):
    X=np.array([x,y])
    distMat=ssp.distance.pdist(X.T)
    sqdist=ssp.distance.squareform(distMat)   #按照distMat的下三角线的元素一列一列凑成一个array
    return sqdist

"可视化"
def draw(solutionbest,locations):
    #路线图绘制
    plt.figure(1)
    for path in solutionbest:
        plt.plot(locations[path][:,0],locations[path][:,1], marker='o')           
    plt.show()


"LK算子优化有回路的TSP问题"
def LK(distance_matrix):
    distance = distance_matrix.copy()
    distance *= 100000   # 将其转为整数，可视为单位行驶成本
    route = elkai.solve_int_matrix(distance)
    route.append(0)  # 最终的解决方案少一个终点0，手动补上
    return route


"计算某条路径的总里程"
def cal_route_cost(route,distance_matrix):
    return np.sum(distance_matrix[route[:-1],route[1:]])


"计算某仓库总路径的总里程"
def cal_solution_cost(solution,distance_matrix):
    solution_cost = 0
    for route in solution:
        solution_cost += cal_route_cost(route,distance_matrix)
    return solution_cost


"生成demo"
def generate_demo(config):
    #在一定范围内随机生成客户位置，其中第一行默认为depot信息
    locations = np.random.uniform(size=(config.customers_num+1, 2))
    locations[:,0] *= (abs(config.map_x_max) + abs(config.map_x_min))
    locations[:,1] *= (abs(config.map_y_max) + abs(config.map_y_min))
    locations[:,0] -= abs(config.map_x_min)
    locations[:,1] -= abs(config.map_y_min) 
    demands = np.random.randint(config.min_demand,config.max_demand+1,size=config.customers_num+1)
    demands[0] = 0
    return locations,demands


