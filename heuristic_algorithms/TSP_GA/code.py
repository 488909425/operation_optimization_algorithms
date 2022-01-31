import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import random
import time

start = time.time()

matplotlib.rcParams['font.family'] = 'STSong'

# 数据载入
city_name = []
city_condition = []
with open('data.txt','r',encoding='UTF-8') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split('\n')[0]
        line = line.split(',')
        city_name.append(line[0])
        city_condition.append([float(line[1]), float(line[2])])
city_condition = np.array(city_condition)        #这里转成array数组对象方便后续的操作，列表不易操作


# 距离矩阵
city_count = len(city_name)
Distance = np.zeros([city_count, city_count])
for i in range(city_count):
    for j in range(city_count):
        Distance[i][j] = math.sqrt(
            (city_condition[i][0] - city_condition[j][0]) ** 2 + (city_condition[i][1] - city_condition[j][1]) ** 2)
# 种群数
count = 200
# 改良次数
improve_count = 500
# 进化次数
iteration = 200
# 设置强者的定义概率，即种群前20%为强者
retain_rate = 0.2
# 变异率
mutation_rate = 0.1
# 设置起点
index = [i for i in range(city_count)]

#总距离
def get_total_distance(path_new):   #path_new是一个路径索引列表
    distance = 0
    for i in range(city_count - 1):
        # count为30，意味着回到了开始的点，此时的值应该为0.
        distance += Distance[int(path_new[i])][int(path_new[i + 1])]
    distance += Distance[int(path_new[-1])][int(path_new[0])]
    return distance

# 改良
#思想：随机生成两个城市，任意交换两个城市的位置，如果总距离减少，就改变染色体。
def improve(x): #x是一个路径索引列表
    i = 0
    distance = get_total_distance(x)
    while i < improve_count:
        # randint [a,b]
        u = random.randint(0, len(x) - 1)
        v = random.randint(0, len(x) - 1)
        if u != v:
            new_x = x.copy()
            ## 随机交叉两个点，t为中间数
            t = new_x[u]
            new_x[u] = new_x[v]
            new_x[v] = t
            new_distance = get_total_distance(new_x)
            if new_distance < distance:
                distance = new_distance
                x = new_x.copy()
        else:
            continue
        i += 1

# 适应度评估，选择，迭代一次选择一次
def selection(population):    #population是多个路径索引列表
    # 对总距离从小到大进行排序
    graded = [[get_total_distance(x), x] for x in population]
    graded = [x[1] for x in sorted(graded)]
    # 选出适应性强的染色体
    retain_length = int(len(graded) * retain_rate)
    #适应度强的集合,直接加入选择中
    parents = graded[:retain_length]
    ## 轮盘赌算法选出K个适应性不强的个体，保证种群的多样性
    s = graded[retain_length:]
    # 挑选的不强的个数
    k = count * 0.2
    # 存储适应度
    a = []
    for i in range(0, len(s)):
        a.append(get_total_distance(s[i]))
    sum = np.sum(a)
    b = np.cumsum(a / sum)           #cumsum是一个累积分布函数，最后一个值为1
    while k > 0:  # 迭代一次选择k条染色体
        t = random.random()
        for h in range(1, len(b)):
            if b[h - 1] < t <= b[h]:
                parents.append(s[h])
                k -= 1
                break
    return parents

# 交叉繁殖
def crossover(parents):
    # 生成子代的个数,以此保证种群稳定
    target_count = count - len(parents)
    # 孩子列表
    children = []
    while len(children) < target_count:
        male_index = random.randint(0, len(parents) - 1)      #random.randint()函数可以取到右端点
        female_index = random.randint(0, len(parents) - 1)
        #在适应度强的中间选择父母染色体
        #下面代码段例子：假设gene1=[4,2],male=[1,3,4,2,5],gen2=[2,1],female=[4,3,2,1,5],则最后child1=[3,4,2,1,5],child2=[3,1,4,2,5]
        if male_index != female_index:
            male = parents[male_index]
            female = parents[female_index]

            left = random.randint(0, len(male) - 2)
            right = random.randint(left + 1, len(male) - 1)

            # 交叉片段
            gene1 = male[left:right]
            gene2 = female[left:right]

            #得到原序列通过改变序列的染色体，并复制出来备用。
            child1_c = male[right:] + male[:right]
            child2_c = female[right:] + female[:right]
            child1 = child1_c.copy()
            child2 = child2_c.copy()

            #已经改变的序列=>去掉交叉片段后的序列
            for o in gene2:
                child1_c.remove(o)
            for o in gene1:
                child2_c.remove(o)

            #交换交叉片段
            child1[left:right] = gene2
            child2[left:right] = gene1

            child1[right:] = child1_c[0:len(child1) - right]
            child1[:left] = child1_c[len(child1) - right:]

            child2[right:] = child2_c[0:len(child1) - right]
            child2[:left] = child2_c[len(child1) - right:]

            children.append(child1)
            children.append(child2)

    return children

# 变异
def mutation(children):
    #children现在包括交叉和优质的染色体
    for i in range(len(children)):
        if random.random() < mutation_rate:
            child = children[i]
            #产生随机数
            u = random.randint(0, len(child) - 4)
            v = random.randint(u + 1, len(child) - 3)
            w = random.randint(v + 1, len(child) - 2)
            child = child[0:u] + child[v:w] + child[u:v] + child[w:]
            children[i] = child
    return children


# 得到最佳纯输出结果
def get_result(population):
    graded = [[get_total_distance(x), x] for x in population]
    graded = sorted(graded)
    return graded[0][0], graded[0][1]


# 使用改良圈算法初始化种群
population = []
for i in range(count):
    # 随机生成个体
    x = index.copy()
    #随机排序
    random.shuffle(x)
    improve(x)
    population.append(x)

#主函数：
register = []
i = 0
distance, result_path = get_result(population)
register.append(distance)
while i < iteration:
    # 选择繁殖个体群
    parents = selection(population)
    # 交叉繁殖
    children = crossover(parents)
    # 变异操作
    children = mutation(children)
    # 更新种群
    population = parents + children
    distance, result_path = get_result(population)
    register.append(distance)
    i = i + 1

print("迭代",iteration,"次后，最优值是：",distance)
print("最优路径：",result_path)
end = time.time()
print("Time used:",end - start)

## 路线图绘制
X = []
Y = []
for index in result_path:
    X.append(city_condition[index, 0])
    Y.append(city_condition[index, 1])
X.append(X[0])
Y.append(Y[0])
#画图
fig = plt.figure()
ax2 = fig.add_subplot()
ax2.set_title('最佳轨迹图')
for i in range(len(x)):
    plt.annotate(result_path[i], xy = (X[i], Y[i]), xytext = (X[i]+0.3, Y[i]+0.3))
plt.plot(X, Y, '-o')
plt.xlabel('经度')
plt.ylabel('纬度')
plt.show()

## 距离迭代图
fig = plt.figure()
ax3 = fig.add_subplot()
ax3.set_title('距离迭代图')
plt.plot(list(range(len(register))), register)
plt.xlabel('迭代次数')
plt.ylabel('距离值')
plt.show()
