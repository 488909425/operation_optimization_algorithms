#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
import scipy.spatial as ssp
import math
import argparse
import elkai
import matplotlib.pyplot as plt
import cv2 
from utils import get_config,genDistanceMat,LK,draw,cal_solution_cost,generate_demo,cal_route_cost
import copy
import matplotlib as mpl    
mpl.rcParams['font.sans-serif'] = ['SimHei']  #指定默认字体
mpl.rcParams['axes.unicode_minus'] = False   #正常显示图像中的负号


# In[2]:


"""
扫描算法：传入当前需要用到的各节点信息，depot信息放在首位
"""
def scanning_method(distance,locations,demands):
    "首先根据depot位置计算各客户与其的夹角"
    temp_loca = locations.copy()
    temp_loca = temp_loca - temp_loca[0]                 # 将仓库作为坐标轴原点
    x = temp_loca[:,0].astype(np.float)
    y = temp_loca[:,1].astype(np.float)
    angle = cv2.cartToPolar(x,y,angleInDegrees=True)[1] # 计算极坐标,提取角度，角度范围是0~360，第0列是向量值大小
    "将角度和各节点索引拼接，然后剔除0点信息"
    concat = np.array(np.hstack((angle,np.matrix(np.arange(locations.shape[0])).T)))[1:]
    concat = concat[np.argsort(concat[:,0])]             # 按角度大小排序
    "计算各节点角度差"
    temp_a = concat[:,0][1:] - concat[:,0][:-1]
    temp_b = 360-concat[-1,0] + concat[0,0]               # 最后一客户和第一个客户的角度差
    angle_difference = np.hstack((temp_a,temp_b))
    max_diff_index = np.argmax(angle_difference)
    temp_list = concat[:,1].astype(np.int).tolist()
    if max_diff_index == angle_difference.shape[0]-1:     # 最大索引为最后一个节点
        "从第一个客户开始扫描"
        scan_index = temp_list       
    else:
        "需要从索引的后一个开始，因为找到的是最大角度，所以需要从参与的两个客户的后一个客户开始逆时针扫描"
        scan_index = temp_list[max_diff_index+1:] + temp_list[:max_diff_index+1]       
    allocation_list = [[0]]
    cur_load = 0
    for i_node in scan_index:
        if cur_load + demands[i_node] < config.capacity + config.EPSILON:    # 保证不违约，消除浮点数的影响
            cur_load += demands[i_node]
            allocation_list[-1].append(i_node)
        else:                                                                 # 违反容量约束
            cur_load = 0
            allocation_list.append([0,i_node])
            cur_load += demands[i_node]            
    "对分配的结果通过LK算子进行优化"
    solution = []
    for path in allocation_list:
        if len(path) > 4:                                      # 长度不大于4的没必要优化
            temp_route = np.array(path)
            solution.append(temp_route[LK(distance[temp_route][:,temp_route])].tolist())
        else:
            path.append(0)
            solution.append(path)
    return solution


# In[3]:


"破坏算子(3个)"
def destory_operators(locations,demands,distance_matrix,solution,destroy_serial_num,destroy_num):
    destroy_list = []
    "------随机移除客户点------"
    if destroy_serial_num == 0:
        for _ in range(destroy_num):                          # 对需要删除的数量进行循环
            while True:
                "防止仓库没有路径"
                index_path = np.random.randint(len(solution))   # 随机该depot的一条路径索引
                if len(solution[index_path]) == 3:             # 该路径只有一个客户
                    destroy_list.append(solution[index_path][1])# 取出该路径的唯一客户放入移除list中
                    del solution[index_path]                   # 删除该路径
                    break
                else: 
                    index_node = np.random.randint(1,len(solution[index_path])-1) # 随机该depot的一条路径索引
                    destroy_list.append(solution[index_path][index_node])
                    del solution[index_path][index_node]
                    break


    "------最坏距离移除节点+扰动------"
    if destroy_serial_num == 1:
        "首先构造一个numpy数组，用于储存 节约度、path索引、path中的位置索引，所以数组形状为 (客户总数，3)"
        save_matrix = np.zeros((locations.shape[0],3))
        "接下来对所有path进行循环，计算其中每个节点的节约度"
        for index_path,path in enumerate(solution):
            for index_node in range(1,len(path)-1):
                "这里节约度计算的是移除后减去移除前，此时节约度均为负，所以最后进行排序就是从小到大排序"
                save_matrix[path[index_node]][0] = (distance_matrix[path[index_node-1]][path[index_node+1]] -                     distance_matrix[path[index_node-1]][path[index_node]] - distance_matrix[path[index_node]][path[index_node+1]])*np.random.uniform(0.7,1)#计算节约度+扰动
                save_matrix[path[index_node]][1] = index_path         # 该仓库内path序号
                save_matrix[path[index_node]][2] = index_node         # 该客户点在path中的位置
        save_matrix = save_matrix[1:,:]                               # 去除第一行，因为第一行为客户点自身
        "对节约度表进行排序,按第一列节约度对行排序（去除第一行）,然后选择需要的前config.destroy_num行数据"
        save_matrix = save_matrix[save_matrix[:,0].argsort()]
        "寻找待移除的点"
        destroy_index = []
        for i_node in range(save_matrix.shape[0]):
            destroy_index.append(i_node)
            if len(destroy_index) == destroy_num:
                break
        save_matrix = save_matrix[destroy_index,:]
        #print(save_matrix)
        "np.lexsort为根据优先级从小到大排序，输入越靠后优先级越高，排序后通过[::-1]来颠倒数组"
        save_matrix = save_matrix[np.lexsort((save_matrix[:,2],save_matrix[:,1]))][::-1]
        "执行删除操作"
        for index in range(len(destroy_index)):                      # 对需要删除的数量进行循环
            if len(solution[int(save_matrix[index][1])]) == 3:       # 该路径只有一个客户
                destroy_list.append(solution[int(save_matrix[index][1])][1])# 取出该路径的唯一客户放入移除list中
                del solution[int(save_matrix[index][1])]             # 删除该路径 
            else:
                destroy_list.append(solution[int(save_matrix[index][1])][int(save_matrix[index][2])]) # 取出该路径的对应位置的客户点放入移除list中
                del solution[int(save_matrix[index][1])][int(save_matrix[index][2])]
                    


    "------移除总里程最大路径+扰动------"
    if destroy_serial_num == 2:
        path_index = 0
        max_dis = 0
        for j_index in range(len(solution)):
            path_dis = cal_route_cost(solution[j_index],distance_matrix)*np.random.uniform(0.7,1)
            if path_dis > max_dis:
                max_dis = path_dis
                path_index = j_index
        destroy_list = solution[path_index][1:-1]
        del solution[path_index]

    return solution,destroy_list


# In[4]:


"修复算子(2个),在修复过程中保证路径可行"
def repair_operators(locations,demands,distance_matrix,solution,destroy_list,repair_serial_num):
    "------贪婪插入------"
    if repair_serial_num == 0:
        for node in destroy_list:          # 对每个待插入点进行循环
            temp_value = np.inf            # 设为一个非常大的数#插入点后增加的里程
            index_path = 0                 # 记录路径顺序
            index_node = 0                 # 记录节点插入位置
            new_construction = False      # 判断是否新增路径
            "首先判断新增路径的增加量"
            temp0 = distance_matrix[0][node]+distance_matrix[node][0] # 计算新增路径的增加量
            if temp0 < temp_value:
                temp_value = temp0
                new_construction = True
            for i_path in range(len(solution)):                      # 对该仓库每条路径循环
                if np.sum(demands[solution[i_path]]) + demands[node] > config.capacity + config.EPSILON: # 载重违约
                    continue
                else:
                    for i_node in range(1,len(solution[i_path])):
                        temp0 = distance_matrix[solution[i_path][i_node-1]][node] +                             distance_matrix[node][solution[i_path][i_node]] -                                 distance_matrix[solution[i_path][i_node-1]][solution[i_path][i_node]]
                        if temp0 < temp_value:
                            temp_value = temp0
                            new_construction = False
                            index_path = i_path
                            index_node = i_node
                
            "进行插入操作"
            if new_construction:
                path = [0,node,0]
                solution.append(path)
            else:
                solution[index_path].insert(index_node,node)                
                
                
    "------贪婪+扰动插入------"
    if repair_serial_num == 1:
        for node in destroy_list:       # 对每个待插入点进行循环
            temp_value = np.inf          # 设为一个非常大的数#插入点后增加的里程
            index_path = 0              # 记录路径顺序
            index_node = 0              # 记录节点插入位置
            new_construction = False    # 这个变量是判断是否新增路径
            "首先判断新增路径的增加量"
            temp0 = (distance_matrix[0][node]+distance_matrix[node][0])*np.random.uniform(0.7,1) # 计算新增路径的增加量
            if temp0 < temp_value:
                temp_value = temp0
                new_construction = True
            for i_path in range(len(solution)):             # 对该仓库每条路径循环
                if np.sum(demands[solution[i_path]]) + demands[node] > config.capacity + config.EPSILON: # 载重违约
                    continue
                else:
                    for i_node in range(1,len(solution[i_path])):
                        temp0 = (distance_matrix[solution[i_path][i_node-1]][node] +                             distance_matrix[node][solution[i_path][i_node]] -                                 distance_matrix[solution[i_path][i_node-1]][solution[i_path][i_node]])*np.random.uniform(0.7,1)
                        if temp0 < temp_value:
                            temp_value = temp0
                            new_construction = False
                            index_path = i_path
                            index_node = i_node                
            "进行插入操作"
            if new_construction:
                path = [0,node,0]
                solution.append(path)
            else:
                solution[index_path].insert(index_node,node)                       
                
  
    return solution


# In[5]:


def ALNS(distance_matrix,locations,demands):
    initial_solution = scanning_method(distance_matrix,locations,demands)        # 通过扫描算法生成初始解
    initial_cost = cal_solution_cost(initial_solution,distance_matrix)           # 计算初始解成本 
    print("initial_solution:",initial_solution)
    draw(initial_solution,locations)
    print("initial_cost",initial_cost)
    "将初始解赋值给当前解及当前最优解"
    current_solution = copy.deepcopy(initial_solution)
    current_cost = initial_cost    
    best_solution = copy.deepcopy(initial_solution)
    best_cost = initial_cost   
    destroy_weight = np.array([10]*config.destroy_size,dtype = np.float64)         # 破坏算子权重
    repair_weight = np.array([10]*config.repair_size,dtype = np.float64)           # 修复算子权重
    score_destroy = np.array([0]*config.destroy_size)                              # 记录destory算子分数
    score_repair = np.array([0]*config.repair_size)                                # 记录repair算子分数 
    time_destroy = np.array([0]*config.destroy_size)                               # 记录destory算子选中次数
    time_repair = np.array([0]*config.repair_size)                                 # 记录repair算子选中次数
    destroy_num = min(max(3,int(config.customers_num/5)),10)                       # 需要调整的破坏节点数量
    
    cost_list = [initial_cost]                                                     # 记录迭代过程中的成本变化
    
    # 模拟退火接受原则
    T_start = 30                   # 初始温度
    no_improve = 0                 # 当前最优解未改进迭代次数
    current_T = T_start            # 设置初始温度    
    "开始ALNS迭代过程"
    for i_iterations in range(10000):
        "破坏过程，按权值概率选择算子，并进行破坏"
        if len(current_solution) == 1:                   # 只有一条路径就不能选择路径移除
            destroy_serial_num = np.random.choice(config.destroy_size-1,p=(destroy_weight[:-1]/sum(destroy_weight[:-1]))) # 选择破坏算子
        else:
            destroy_serial_num = np.random.choice(config.destroy_size,p=(destroy_weight/sum(destroy_weight)))
        time_destroy[destroy_serial_num] += 1          # 记录算子选择次数
        temp_solution,destroy_list = destory_operators(locations,demands,distance_matrix,copy.deepcopy(current_solution),destroy_serial_num,destroy_num)#对当前解进行破坏   
        "修复过程，按权值概率选择算子，并进行修复"
        repair_serial_num = np.random.choice(config.repair_size,p=(repair_weight/sum(repair_weight))) # 选择修复算子    
        time_repair[repair_serial_num] += 1
        new_solution = repair_operators(locations,demands,distance_matrix,temp_solution,destroy_list,repair_serial_num)
        new_cost = cal_solution_cost(new_solution,distance_matrix) # 计算新解适应度        
        "比较适应度并更新解"
        if new_cost + config.EPSILON < best_cost:
            no_improve = 0
            new_cost = cal_solution_cost(new_solution,distance_matrix)#计算新解适应度
            best_cost = new_cost
            print("第%s次迭代更新最优解成本为:"%(i_iterations+1),best_cost)
            current_cost = new_cost
            
            cost_list.append(current_cost)
            
            "更新路径"
            best_solution = copy.deepcopy(new_solution)
            current_solution = copy.deepcopy(new_solution)
            "加分"
            score_destroy[destroy_serial_num] += 10
            score_repair[repair_serial_num] += 10        
        else:
            no_improve += 1
            "这里不包含新解==当前解的情况"
            if new_cost + config.EPSILON < current_cost:  
                current_cost = new_cost                  
                current_solution = copy.deepcopy(new_solution)
                
                cost_list.append(current_cost)
                
                score_destroy[destroy_serial_num] += 5
                score_repair[repair_serial_num] += 5        
            else:        
                "计算接受概率，current_fitness-new_fitness通常为负，但这里也可能包含相减为0的情况"
                P_S_T = np.exp((current_cost-new_cost)/current_T)
                if np.random.uniform() <= P_S_T:   #接受                    
                    current_cost = new_cost                  
                    current_solution = copy.deepcopy(new_solution)
                    
                    cost_list.append(current_cost)
                    
                    score_destroy[destroy_serial_num] += 3
                    score_repair[repair_serial_num] += 3          
        

        "---------更新权值概率---------"
        if (i_iterations+1) % int(20) == 0:   # 到达权值更新次数 
            for f in range(config.destroy_size):
                if time_destroy[f] == 0:
                    destroy_weight[f] *= (1-config.ρ)
                else:
                    destroy_weight[f] = config.ρ*score_destroy[f]/time_destroy[f] + (1-config.ρ)*destroy_weight[f]   
                
            for f in range(config.repair_size):
                if time_repair[f] == 0:
                    repair_weight[f] *= (1-config.ρ)
                else:
                    repair_weight[f] = config.ρ*score_repair[f]/time_repair[f] + (1-config.ρ)*repair_weight[f]           
        
            "重新初始化各变量"
            score_destroy = np.array([0]*config.destroy_size)  # 记录destory算子分数
            score_repair = np.array([0]*config.repair_size)    # 记录repair算子分数 
            time_destroy = np.array([0]*config.destroy_size)   # 记录destory算子选中次数
            time_repair = np.array([0]*config.repair_size)     # 记录repair算子选中次数        
        
        if no_improve == 200:           # 达到最大未改进次数，返回结果
            return best_solution,best_cost,cost_list
        
        current_T *= 0.94               # 温度衰减
        
        "判断是否需要再加热"
        if current_T < 0.1:
            current_T = T_start         # 初始化温度 
    
    return best_solution,best_cost,cost_list


# In[6]:


if __name__ == '__main__':
    np.random.seed(2)           # 固定随机种子，使得每次运行结果相同
    config = get_config()       # 参数实例化
    locations,demands = generate_demo(config)  # 生成demo
    distance_matrix = genDistanceMat(locations[:,0], locations[:,1])  # 计算距离矩阵
    best_solution,best_cost,cost_list = ALNS(distance_matrix,locations,demands)
    print("最终最优解方案为:",best_solution,"解成本为:",best_cost)
    draw(best_solution,locations)
    plt.figure(2)
    plt.plot(cost_list)
    plt.title('迭代过程')


# In[ ]:




