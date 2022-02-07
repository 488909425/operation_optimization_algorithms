import numpy as np

#写入原始评分矩阵
#np.mat索引方式不同于np.array
scoreData=np.mat([
[5,2,1,4,0,0,2,4,0,0,0],
[0,0,0,0,0,0,0,0,0,3,0],
[1,0,5,2,0,0,3,0,3,0,1],
[0,5,0,0,4,0,1,0,0,0,0],
[0,0,0,0,0,4,0,0,0,4,0],
[0,0,1,0,0,0,1,0,0,5,0],
[5,0,2,4,2,1,0,3,0,1,0],
[0,4,0,0,5,4,0,0,0,0,5],
[0,0,0,0,0,0,4,0,4,5,0],
[0,0,0,4,0,0,1,5,0,0,0],
[0,0,0,0,4,5,0,0,0,0,3],
[4,2,1,4,0,0,2,4,0,0,0],
[0,1,4,1,2,1,5,0,5,0,0],
[0,0,0,0,0,4,0,0,0,4,0],
[2,5,0,0,4,0,0,0,0,0,0],
[5,0,0,0,0,0,0,4,2,0,0],
[0,2,4,0,4,3,4,0,0,0,0],
[0,3,5,1,0,0,4,1,0,0,0]])

print(scoreData)
print(np.shape(scoreData))

#定义计算余弦近似度函数
def cosSim(vec_1,vec_2):
	dotProd=float(np.dot(vec_1.T,vec_2)) #对于一维向量，实际上有无转置不影响
	normProd=np.linalg.norm(vec_1)*np.linalg.norm(vec_2)
	return 0.5+0.5*(dotProd/normProd)  #转为[0,1]区间
	
	
#定义计算用户对某个菜品的评价分数函数
#scoreDataRC为行压缩矩阵
def estScore(scoreData,scoreDataRC,userIndex,itemIndex):
	n=np.shape(scoreData)[1]     #列数，菜品数量
	simSum=0
	simSumScore=0
	for i in range(n):
		userScore=scoreData[userIndex,i]   #用户对菜品的评价分数
		if userScore==0 or i==itemIndex:   #0表示不评价，userindex是目标预测用户,i==itemIndex是目标预测菜品
			continue
		sim=cosSim(scoreDataRC[:,i],scoreDataRC[:,itemIndex]) #利用上面定义的函数计算相似度
		simSum=float(simSum+sim)
		simSumScore+=userScore*sim
	if simSum==0:
		return 0
	return simSumScore/simSum
	
#进行SVD分解
U,sigma,VT=np.linalg.svd(scoreData)                 #sigma为向量array形式
print(np.shape(U),np.shape(sigma),np.shape(VT))


sigmaSum=0
k_num=0

#确定要取的特征值的数量
for k in range(len(sigma)):
	#奇异值的平方即为特征值
	sigmaSum+=sigma[k]**2
	if float(sigmaSum)/float(np.sum(sigma**2))>0.9:    #取占特征值90%的前n个
		k_num=k+1
		break
		
sigma_k=np.mat(np.eye(k_num)*sigma[:k_num])
scoreDataRC=np.dot(np.dot(sigma_k,U.T[:k_num,:]),scoreData)  #对原始矩阵进行行压缩

n=np.shape(scoreData)[1]

userIndex=17  #取第18个用户

for i in range(n):
	userScore=scoreData[17,i]
	if userScore!=0:
		continue
	print("用户对第{0}个菜品的预估分数为{1}".format(i+1,estScore(scoreData,scoreDataRC,userIndex,i)))
