# **代码介绍**

>***这是一个贝叶斯算法库，包含三个类，分别为评分函数类、模型训练类以及模型调用类。***

---

## **１. 贝叶斯网基本原理介绍**

>贝叶斯网络是由贝叶斯公式引出的，朴素贝叶斯、半朴素贝叶斯均是贝叶斯网络的特例，下文着重探讨贝叶斯网络的推导过程以及计算过程，其余相关内容可以在查询相关论文进行进一步的学习。
>### １.1 贝叶斯公式
>
>>谈到贝叶斯公式就不得不提到统计学中的贝叶斯派与频率派，两者从不同的观察角度出发，因此建模的角度会有所不同，具体的不同可以上网查询，这里只说结论：频率派对事件本身建模，估计建模所需参数；而贝叶斯派认为建模所估计的参数的参数空间是非常重要的。举个例子，频率派的参数估计常用的是极大似然估计(MLE)，而贝叶斯派是贝叶斯估计，具体的公式如下：
>>>极大似然估计：
>>>$$
>>>\theta_{MLE}=argmaxP(X|\theta)
>>>$$
>>>贝叶斯估计：
>>>$$
>>>P(\theta|X)=\frac {P(X|\theta)P(\theta)}{P(X)}\propto P(X|\theta)P(\theta)
>>>$$
>>
>>由公式可以看出，极大似然估计是寻找使得事件发生概率最大的参数$\theta$，贝叶斯派则是在考虑事件发生概率的同时，亦观察参数$\theta$的先验概率$P(\theta)$即$\theta$的参数空间。
>>
>>举个例子：掷一个硬币10次，向上的次数是７次，按照极大似然估计的计算方法，预测得到的$\theta$是二项分布的$p$值（正面向上的概率）为0.7，但是这个和我们的常识不符；如果我们知道$p$的先验分布（即先验知识：我们知道$p=0.5$的概率比$p=0.7$的概率高），那么最后的预测得到的$p$会比0.7小，更接近0.5。所以我们知道在知道先验分布的情况下，事件的随机扰动会被先验概率所平衡掉。
>
>### １.2 贝叶斯网的计算原理
>
>>贝叶斯网的网络结构是由网络节点和有向边组成，网络中不存在有向环结构；每个网络节点代表一个特征；有向边存在于两个节点之间，由父节点指向子节点，由此构成一个贝叶斯网络，下文的实例环节有一个贝叶斯网络的图像。接着就由下文的贝叶斯网络来展示贝叶斯网络的计算方法。
>>>贝叶斯网络的计算方法：
>>>$$
>>>P(Y|touch,color,umbilicus,root,grain,sound)\\
>>>=\frac {P(Y,touch,color,umbilicus,root,grain,sound)}{P(touch,color,umbilicus,root,grain,sound)}\\
>>>=\frac {P(Y)P(touch|Y,grain)P(color|Y,grain)P(umbilicus|Y,grain)P(root|umbilicus,Y)P(sound|root,Y)P(grain|Y)}{P(touch,color,umbilicus,root,grain,sound)}\\
>>>\propto {P(Y)P(touch|Y,grain)P(color|Y,grain)P(umbilicus|Y,grain)P(root|umbilicus,Y)P(sound|root,Y)P(grain|Y)}
>>>$$
>>
>>由上述公式可以知道当我们知道除了$Y$以外的其他特征时，可以通过贝叶斯公式预测得到$Y$的各类值的概率。
>
>一个含有n个特征节点的贝叶斯网络的结构有超过$n!$种。因此想要找到最优的网络结构是一个NP问题，对此为了找到最优的网络结构，需要设计评分函数来给网络结构评分。网络评分分为两部分，一部分是模型的拟合能力，还有一部分是正则惩罚项，避免网络结构过于复杂。模型的训练过程就是使得网络评分不断上升，以找到较优的网络结构，具体内容在下文会一一展示。

---

## **２. Net_score（评分函数）**

***评分函数类内置了两个评分函数，BIC以及BCPS函数。***

>### 函数公式
>
>* BCPS
>$$
>BCPS(S|D)=\sum_{i=1}^n\sum_{j=1}^{q_i}\sum_{k=1}^{r_i}m_{ijk}\theta_{ijk}-\lambda\sum_{i=1}^{n}q_i(r_i-1)m
>$$
>* BIC
>$$
>BIC(S|D)=\sum_{i=1}^n\sum_{j=1}^{q_i}\sum_{k=1}^{r_i}m_{ijk}log({\theta_{ijk}})-\lambda\sum_{i=1}^{n}q_i(r_i-1)log(m)
>$$
>其中$n$是网络节点数量；$q_i$是节点$i$的父节点组合数量；$r_i$是节点$i$的类别数量；$m_{ij}$是节点$i$的父类组合为时的数据量；$m_{ijk}$是节点$i$的父类组合为$j$时，节点$i$取值为$k$时的数据量；$\theta_{ijk}=m_{ijk}/m_{ij}$为节点$i$的条件概率；$\lambda$是正则项系数。

---

## **３. Bayes_Net（网络训练）**

***网络训练包含两大部分：初始结构的生成以及基于现有网络结构的训练。***
>### **３.1  初始结构**
>
>***选择初始结构的方式有很多种，这里选择以TAN半朴素贝叶斯(Tree Augmented naive Bayes)[Friedman et al.,1997]作为初始网络结构。***
>>**Tan网络结构的生成步骤：**
>>
>><1>　计算任意两个属性间的条件互信息；
>>$$
>I(x_i,x_j|y)=\sum_{x_i,x_j;c\in y}P(x_i,x_j|c)log\frac {P(x_i,x_j|c)}{P(x_i|c)P(x_i|c)}
>$$
>>
>><2>　以所有属性节点以及节点间的权重构建最大带权生成树，任意两个节点间的权重为$I(x_i,x_j|y)$;
>>
>><3>　加入$Y$节点，使其作为所有属性节点的父节点。

>---

>>**对应代码步骤：**
>>
>><1>　载入数据（数据均在data文件夹下）：
>>```python
>>##这里以西瓜数据为例
>>import pandas as pd
>>data = pd.read_csv('./melon.csv')
>>```
>>data格式如下(均需为int格式):
>>|Y|color|...|touch|
>>|:-:|:-:|:-:|:-:|
>>|1|2|...|2|
>>|1|3|...|2|
>>|...|...|...|...|
>>|0|3|...|1|

>><2>　将data传入算法库，使其实例化:
>>```python
>>import sys
>>sys.path.append('包存放路径')
>>import BayesNet as BN
>>bayesian = BN.Bayes_Net(data)   
>>```

>><3>　计算条件互信息矩阵：
>>```  python
>>##计算以Y为条件的条件互信息矩阵
>>cmi_arr = bayesian.CMI_ARR('Y')    
>>```
>>矩阵形式如下:
>>|cmi|Y|color|root|...|touch|
>>|:-:|:-:|:-:|:-:|:-:|:-:|
>>|Y|0.0|0.0|0.0|...|0.0|
>>|color|0.0|0.0|0.45|...|0.09|
>>|root|0.0|0.45|0.0|...|0.50|
>>|...|...|...|...|...|...|
>>|touch|0.0|0.09|0.50|...|0.0|

>><4>　以条件互信息作为权重生成最大带权生成树：
>>```python
>> ##将条件互信息作为权重提取出来
>>w = bayesian.W_OUT(cmi_arr)  
>>
>>"""代码为生成最大带权生成树提供了两种方法"""
>> ##Prim算法生成最大带权生成树
>>tree1 = bayesian.Prim_tree(w)    
>> ##Kruskal算法生成最大带权生成树
>>tree2 = bayesian.Kruskal_tree(w)    
>>```
>>最大带权生成树以字典形式储存，形式如下：
>>```python
>>tree1 = {'grain':[],'umbilicus':['grain'],'root':['umbilicus'],'sound':['root'],'touch':['grain'],'color':['grain'],'Y':['grain']} 
>>
>>##属性间的带权生成树本不应该出现Y节点，因为Y与其他节点的权重均为0。但是网络结构中存在Y节点，为了方便之后操作，代码在生成最大带权生成树时会带上Y(计算cmi时的条件节点)
>>```

>><5>　生成TAN半朴素贝叶斯网络作为初始结构：
>>```python
>>##将Y节点作为所有属性节点的父类，Y节点的父类置空
>>for key in tree1.keys():
>>   tree[key].append('Y')
>>tree['Y'] = []    　　　　　           
>>```
>>TAN网络结构如下：
>>```python
>>tree1 = {'grain':['Y'],'umbilicus':['grain', 'Y'],'root':['umbilicus', 'Y'], 'sound':['root', 'Y'],'touch':['grain', 'Y'],'color':['grain', 'Y'],'Y':[]} 
>>```
>>结构展示如下：
>>
>>![TAN结构](https://wx1.sinaimg.cn/mw690/00872OYVly1gdmgyv7gxgj30fy0cmdgd.jpg)

>---

>### **３.2 结构训练**
>
>***代码采用的网络结构训练方法为HCNew爬山法，该方法为在网络中随机选择任意两个节点，如果这两个节点之间不存在边，则在这两个节点间生成一条边；如果两个节点之间存在边，则以$p$的概率删除这条边，以１－$p$的概率旋转这条边。结构改变后则计算新结构的评分函数，如果高于原结构的评分，则保留新的结构，否则还原至原来的结构。***
>
>>代码实现如下：
>>```python
>>new_tree = bayesian.HCNew(father_dict = tree1, lag=50, del_prob = 0.3, lamb = 0.05, score_function = 'BCPS')
>>##father_dict是需要训练的网络结构，lag是训练步数，del_prob是删除边的概率，lamb是正则项系数，score_function是选择的评分函数。
>>```
>>训练出来的结构为：
>>```python
>>tree1 = {'grain':['Y'],'umbilicus':['root'],'root':['sound', 'color', 'touch', 'Y'],'sound':[],'touch':[],'color':[],'Y':[]} 
>>```
>>计算得到各个节点的概率表：
>>```python
>>bayesian.Prob_Graph(new_tree[0])
>>```
>>保存模型：
>>```python
>>bayesian.Save('./model.pkl')
>>```

## **４. Bayes_Predict（模型调用）**

***模型调用主要用于输出需要预测的节点的概率。***

>### **４.1 模型预测**
>
>***选择需要预测的节点，按照上文的西瓜数据，需要预测的是Y节点。***
>
>>代码实现如下：
>>```python
>> ##调用模型，并用西瓜数据来检验拟合效果
>>test_feature = data.iloc[:, 1:]             
>>pred = BN.Bayes_Predict('./model.pkl', test_feature）　　　　
>>prod = pred.Prob(['Y'])
>>```
>>预测的结果为：
>>|编号|0|1|
>>|:-:|:-:|:-:|
>>|1|0.023|0.977|
>>|2|0.026|0.974|
>>|3|0.014|0.986|
>>|4|0.043|0.957|
>>|...|...|...|
>>|16|0.927|0.073|
>>|17|0.758|0.242|
>>
>>比较原label，17个样本有一个预测错误，准确率为：94.12%

***
<p align='right'>Author : Junyu</p>
<p align='right'>date : 2020-04-10</p>

