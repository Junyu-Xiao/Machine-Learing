##Author : Junyu  
##2019-10-18


#载入所需程辑包
import itertools as itools
import pandas as pd  
import numpy as np
import math
import copy
import time
import pickle
import hashlib
from tqdm import tqdm


##网络评分
class Net_score():
    """网络评分函数类，在网络训练时可以随时调用"""
    
    
    
    
    
    def __init__(self, train_data):
        """data是原始数据表, 需要是DataFrame格式"""
        self.train_data = np.array(train_data, dtype = 'int')
        self.col_name = list(train_data.keys())
    
    
    
    
    
    def BCPS(self, son_index, parent_index, lamb = 0.05): 
        """采用BCPS函数计算单个节点评分.
           其中son_index是子类名称索引; parent_index是父类节点名称索引列表, 其格式必须是list.
           lamb是正则项乘数, 其越大, 网络复杂度越低, 过拟合风险越小; 其越大, 网络复杂度越高, 过拟合风险越大."""
        
        ##获取所需信息 
        label_num = len(set(np.asarray(self.train_data[:, son_index]))) ##子类取值个数 ri 
        total_index = parent_index[:]  
        total_index.insert(0, son_index)  
        data_num = self.train_data.shape[0] ##数据长度 m
        
        #各特征类型集合
        total_set = list() 
        
        ##各特征元素集合
        for index in total_index: 
            set_num = tuple(set(np.asarray(self.train_data[:, index]))) 
            total_set.append(set_num) 
        
        ##所有特征可能的组合
        bs = list()
        kind_list = list()
        for kind in itools.product(*total_set): 
            kind_list.append(kind)
            mijk = (np.asarray(self.train_data[:, total_index]) == kind).min(1).sum() ##mijk
            theta = (np.asarray(self.train_data[:, total_index]) == kind).min(1).mean() ##theta
            
            ##记录分数    
            bs.append(mijk * theta)
            
        ##计算节点BCPS函数值    
        total_kind = len(bs)
        parent_kind = total_kind/label_num  #父类组合数目 qi
        bs_value = sum(bs)
        regular = lamb * data_num * parent_kind * (label_num - 1)
        BCPS_VAL = bs_value - regular
        return BCPS_VAL   
    
    
    
    
    
    def BCPS_TOTAL(self, father_dict, lamb = 0.05):
        """采用BCPS评分函数对整个网络进行评分.
           先使用BCPS对各个节点进行评分, 再求整体评分.
           其中father_dict就是整个网络的结构, 用字典储存, key()是子节点, values()是列表, 其中元素为各个父节点.
           father_dict应为有向无环图."""
        
        ##整体BCPS分数计算
        BCPS_dict = dict()
        for son in self.col_name:
            father_list = father_dict[son]
            son_index = self.col_name.index(son)
            father_index = [self.col_name.index(i) for i in father_list]
            BCPS_value = self.BCPS(son_index, father_index, lamb)
            BCPS_dict[son] = BCPS_value
        return BCPS_dict 
    


    
    def BIC(self, son_index, parent_index, lamb = 0.05): 
        """采用BIC函数计算单个节点评分.
           其中son_index是子类名称索引; parent_index是父类节点名称索引列表, 其格式必须是list.
           lamb是正则项乘数, 其越大, 网络复杂度越低, 过拟合风险越小; 其越大, 网络复杂度越高, 过拟合风险越大."""
        
        ##获取所需信息 
        label_num = len(set(np.asarray(self.train_data[:, son_index]))) ##子类取值个数 ri 
        total_index = parent_index[:]  
        total_index.insert(0, son_index)  
        data_num = self.train_data.shape[0] ##数据长度 m
        
        #各特征类型集合
        total_set = list() 
        
        ##各特征元素集合
        for index in total_index: 
            set_num = tuple(set(np.asarray(self.train_data[:, index]))) 
            total_set.append(set_num) 
        
        ##所有特征可能的组合
        bc = list()
        kind_list = list()
        for kind in itools.product(*total_set): 
            kind_list.append(kind)
            mijk = (np.asarray(self.train_data[:, total_index]) == kind).min(1).sum() ##mijk
            theta = (np.asarray(self.train_data[:, total_index]) == kind).min(1).mean() ##theta
            
            ##记录分数    
            bc.append(mijk * math.log(theta + 1e-50, 2))
            
        ##计算节点BIC函数值    
        total_kind = len(bc)
        parent_kind = total_kind/label_num  #父类组合数目 qi
        bc_value = sum(bc)
        regular = lamb * math.log(data_num, 2) * parent_kind * (label_num - 1)
        BIC_VAL = bc_value - regular
        return BIC_VAL   
    
    
    
    
    def BIC_TOTAL(self, father_dict, lamb = 0.05):
        """采用BIC评分函数对整个网络进行评分.
           先使用BIC对各个节点进行评分, 再求整体评分.
           其中father_dict就是整个网络的结构, 用字典储存, key()是子节点, values()是列表, 其中元素为各个父节点.
           father_dict应为有向无环图."""
        
        ##整体Bic分数计算
        BIC_dict = dict()
        for son in self.col_name:
            father_list = father_dict[son]
            son_index = self.col_name.index(son)
            father_index = [self.col_name.index(i) for i in father_list]
            BIC_value = self.BIC(son_index, father_index, lamb)
            BIC_dict[son] = BIC_value
        return BIC_dict 
	
	
    
    
    
    def Score_F(self, son_index, parent_index, lamb = 0.05, score_function = 'BCPS'):
        if score_function == 'BCPS':
            return self.BCPS(son_index, parent_index, lamb)
        elif score_function == 'BIC':
            return self.BIC(son_index, parent_index, lamb)
        
        
        
    def Score_T(self, father_dict, lamb = 0.05, score_function = 'BCPS'):
        if score_function == 'BCPS':
            return self.BCPS_TOTAL(father_dict, lamb)
        elif score_function == 'BIC':
            return self.BIC_TOTAL(father_dict, lamb)
    
    
    
    

class Bayes_Net(Net_score):
    """用于贝叶斯网络结构训练,
       贝叶斯表生成"""
    
    
    
    
    
    def __init__(self, train_data):
        """data是原始数据表, 需要是DataFrame格式"""
        self.train_data = np.array(train_data, dtype = 'int')
        self.col_name = list(train_data.keys()) 
        
      


    
    ##条件互信息
    def condition_mutual_info(self, index1, index2, condition_index): 
        """计算个特征之间互信息.
           series1 和 series2 是两个特征的序列, 应使用pd.Series格式."""
        
        ##获取数据长度以及两个特征的元素集合，并将两列特征合并
        total_index = [index1, index2, condition_index]
        set1 = list(set(np.asarray(self.train_data[:, index1]))) 
        set2 = list(set(np.asarray(self.train_data[:, index2]))) 
        condition_set = list(set(np.asarray(self.train_data[:, condition_index]))) 
        
        ##计算两个特征之间的互信息
        CMI_list = list()
        for kind in itools.product(set1, set2, condition_set):
            total_num = (np.asarray(self.train_data[:, [condition_index]]) == [kind[2]]).min(1).sum()
            p12 = (np.asarray(self.train_data[:, [index1, index2, condition_index]]) == kind).min(1).sum()/total_num
            p1 = (np.asarray(self.train_data[:, [index1, condition_index]]) == [kind[0], kind[2]]).min(1).sum()/total_num
            p2 = (np.asarray(self.train_data[:, [index2, condition_index]]) == [kind[1], kind[2]]).min(1).sum()/total_num
            if p12 == 0:
                CMI_list.append(0)
            else:
                cmi = p12 * math.log(p12/(p1 * p2), 2)
                CMI_list.append(cmi) 
        CMI = sum(CMI_list)
        return CMI
            
            
        
        
    ##条件互信息矩阵    
    def CMI_ARR(self, condition_col):
        """计算出任意两个特征的互信息, 并生成矩阵"""
        
        cmi_arr = pd.DataFrame(0, index = self.col_name, columns = self.col_name)
        condition_index = self.col_name.index(condition_col)
        for i in tqdm(self.col_name):
            col_new = [j for j in self.col_name if j != i] 
            for j in col_new:
                index1 = self.col_name.index(i)
                index2 = self.col_name.index(j)
                cmi = self.condition_mutual_info(index1, index2, condition_index)
                cmi_arr.loc[i, j] = cmi
        self.cmi_arr = cmi_arr
        return cmi_arr 
    


	 
        
    ##互信息
    def mutual_info(self, index1, index2): 
        """计算个特征之间互信息.
           series1 和 series2 是两个特征的序列, 应使用pd.Series格式."""
        
        ##获取数据长度以及两个特征的元素集合，并将两列特征合并
        total_index = [index1, index2]
        set1 = list(set(np.asarray(self.train_data[:, index1]))) 
        set2 = list(set(np.asarray(self.train_data[:, index2]))) 
        
        ##计算两个特征之间的互信息
        MI_list = list()
        for kind in itools.product(set1,set2):
            p12 = (np.asarray(self.train_data[:, total_index]) == kind).min(1).mean()
            p1 = (np.asarray(self.train_data[:, index1]) == kind[0]).mean()
            p2 = (np.asarray(self.train_data[:, index2]) == kind[1]).mean()
            if p12 == 0:
                MI_list.append(0)
            else:
                mi = p12 * math.log(p12/(p1 * p2), 2)
                MI_list.append(mi) 
        MI = sum(MI_list)
        return MI
            
            
        
        
    ##互信息矩阵    
    def MI_ARR(self):
        """计算出任意两个特征的互信息, 并生成矩阵"""
        
        mi_arr = pd.DataFrame(0, index = self.col_name, columns = self.col_name)
        for i in tqdm(self.col_name):
            col_new = [j for j in self.col_name if j != i] 
            for j in col_new:
                index1 = self.col_name.index(i)
                index2 = self.col_name.index(j)
                mi = self.mutual_info(index1, index2)
                mi_arr.loc[i, j] = mi
        self.arr = mi_arr
        return mi_arr 
    
    
    
    
    ##权重提取
    def W_OUT(self, mi_arr):
        """将互信息矩阵中, 任意两个特征之间的互信息提取出来.
           提出的互信息当成权重信息, 用以之后的最大带权生成树的建立."""
        
        w_out = []
        col_name = [col for col in mi_arr]
        for i in tqdm(range(1, len(col_name))):
            for j in range(i):
                index = (col_name[i], col_name[j])
                w = mi_arr.iloc[i, j]
                w_out.append([index, w])
        self.w = w_out
        return w_out
    
    
    
    
    
    ##最大权生成树(Prim算法)
    def Prim_tree(self, w_values, reverse = True):
        """采用Prim算法, 根据节点权重信息, 生成最大带权生成树.
           w_values是一个双层列表, 形如[[(x1, x2), p],...],
           其中含义是: x1与x2节点之间的权重为p. """
        
        ##建立已存在列表（已被使用的树节点）
        exist = []
        
        ##获取节点列表
        ele_set = []
        for i in w_values:
            for j in i[0]:
                ele_set.append(j)
        ele_set = tuple(set(ele_set))
        
        ##令第一个节点作为父节点，其父节点为空集
        exist.append(ele_set[0])
        father_dict = {}
        father_dict[str(ele_set[0])] = []
        
        ##用存在列表中的节点去连接未存在列表中的节点，检索找到转置最大的连接线，并将该连接线上未存在列表中的节点加入到存在列表
        while len(exist) < len(ele_set):
            unexist = [j for j in ele_set if j not in exist]
            index_list = []
            for j in exist:
                for m in unexist:
                    index_num = [n for n in range(len(w_values)) if (w_values[n][0].count(j)>0 and w_values[n][0].count(m)>0)]
                    if len(index_num) > 0:
                        index_list.append(index_num[0])
                    else:
                        continue
            w_list = [w_values[j] for j in index_list]
            w_list = sorted(w_list, key = lambda x:x[1],reverse = reverse)
            new_w = w_list[0][0]
            father_w = [j for j in new_w if j in exist][0]
            son_w = [j for j in new_w if j in unexist][0]
            exist.append(son_w)
            father_dict[str(son_w)] = [father_w]
            
        ##输出树结构
        self.tree = father_dict
        return father_dict
    
    
    
    
    
    ##最大权生成树(Kruskal算法)
    def Kruskal_tree(self, w_values, reverse = True):
        """采用Kruskal算法, 根据节点权重信息, 生成最大带权生成树.
           w_values是一个双层列表, 形如[[(x1, x2), p],...],
           其中含义是: x1与x2节点之间的权重为p. """
        
        ##建立分组字典(用于查并集); 建立树结构字典
        group_dict = {}
        father_dict = {}
        
        ##节点组号初始化; 树结构初始化
        for i in w_values:
            for j in i[0]:
                group_dict[str(j)] = 0
                father_dict[str(j)] = []
                
        ##将权重从大到小排序
        w_sorted = sorted(w_values, key = lambda x:x[1], reverse = reverse)
        exist1 = w_sorted.pop(0)[0]
        father_ele = []
        for i in exist1:
            group_dict[str(i)] = 1
            father_ele.append(i)
        father_dict[str(exist1[1])].append(exist1[0])
        
        ##节点取并集
        i = 2
        index_list = []
        while set(group_dict.values()) != {1} and len(w_sorted) != 0:
            w_index = w_sorted.pop(0)[0] 
            index1, index2 = w_index[0], w_index[1]
            index_group1, index_group2 = group_dict[str(index1)], group_dict[str(index2)]
            
            ##第一个节点有类, 第二个节点无类时, 将第二个节点归入第一个节点的类
            if index_group1 != 0 and index_group2 == 0:
                group_dict[str(index2)] = group_dict[str(index1)]
                index_list.append(w_index)
                
            ##第二个节点有类, 第一个节点无类时, 将第一个节点归入第二个节点的类
            elif index_group1 == 0 and index_group2 != 0:
                group_dict[str(index1)] = group_dict[str(index2)]
                index_list.append(w_index)
                
            ##第一第二节点均无类时, 将第一第二节点归入新的类i
            elif index_group1 == 0 and index_group2 == 0:
                group_dict[str(index1)] = i
                group_dict[str(index2)] = i
                i += 1
                index_list.append(w_index)
                
            ##当第一第二个节点都存在类, 且类相同时, 则不选择这条连接线
            elif index_group1 == index_group2 and index_group1 != 0:
                continue
                
            ##当第二个节点的类存在, 且第一个节点的类大于第二个节点的类(第一个节点的类较新)时, 将第一节点类中所有的节点都归为第二节点的类
            elif index_group1 > index_group2 and index_group2 != 0:
                for k in group_dict.items():
                    if k[1] == index_group1:
                        group_dict[k[0]] = index_group2
                    else:
                        continue
                index_list.append(w_index)
                
            ##当第一个节点的类存在, 且第一个节点的类小于第二个节点的类(第二个节点的类较新)时, 将第二节点类中所有的节点都归为第一节点的类
            elif index_group1 < index_group2 and index_group1 != 0:
                for k in group_dict.items():
                    if k[1] == index_group2:
                        group_dict[k[0]] = index_group1
                    else:
                        continue
                index_list.append(w_index)
        
        ##树结构生成
        while len(index_list) > 0:
            index_ele = index_list.pop(0)
            ele_status = [int(j in father_ele) for j in index_ele]
            if sum(ele_status) == 0:
                index_list.append(index_ele)
            else:
                father_index = ele_status.index(1)
                father_dict[str(index_ele[1-father_index])].append(index_ele[father_index])
                father_ele.append(index_ele[1-father_index])
        
        ##输出树结构
        self.tree = father_dict
        return father_dict
    
    
    
    
    ##环状判别
    def GROUP_DIST(self, father_dict):
        """判断结构是否存在不同集群.
           输出为True时, 不存在不同集群.
           否则输出集群编号及其对应节点."""
        
        ##获取节点名称
        cols = list(father_dict.keys())
        
        ##建立节点关联矩阵
        DAG_ARR = pd.DataFrame(index = cols, columns = cols)
        for key in father_dict.keys(): 
            value = father_dict[key] 
            for origin in value:
                DAG_ARR.loc[origin, key] = 1 
                
        ##建立集群字典
        group = dict()
        group_no = 1
        
        ##初始化集群序号
        for column in DAG_ARR:
            group[column] = 0
            
        ##所属集群检索
        for column in DAG_ARR:
            parent_list = DAG_ARR[column][DAG_ARR[column] == 1].index
            son_list = DAG_ARR.loc[column][DAG_ARR.loc[column] == 1].index
            point_group = group[column]
            
            ##该节点无其他关联节点时, 新增一类
            if len(parent_list) == 0 and len(son_list) == 0:
                group[column] = group_no
                group_no += 1
            
            ##感染算法(越小的集群编号拥有越强的感染能力, 当节点连接在一起时, 所有相连节点会被其中最小集群编号的节点感染为同一集群编号)
            else:
                parent_group = [group[name] for name in parent_list]
                son_group = [group[name] for name in son_list]
                parent_no = [no for no in parent_group if no > 0]
                son_no = [no for no in son_group if no > 0]
                parent_no.extend(son_no)
                connect_no = list(set(parent_no))
                if len(connect_no) == 0:
                    min_no = 0
                else:
                    min_no = min(connect_no)
                
                ##检索节点无集群编号且关联节点也无集群编号时, 所有关联节点的集群编号统一为一个新的集群编号
                if point_group == 0 and min_no == 0:
                    for point in parent_list:
                        group[point] = group_no
                    for point in son_list:
                        group[point] = group_no
                    group_no += 1 
                
                ##检索节点无集群编号且关联节点有集群编号时, 所有相连节点统一为其中最小的集群编号
                elif point_group == 0 and min_no > 0:
                    group[column] = min_no
                    for point in parent_list:
                        group[point] = min_no
                    for point in son_list:
                        group[point] = min_no
                    for item in group.items():
                        if item[1] in connect_no:
                            group[item[0]] = min_no
                        else:
                            continue
                        
                ##检索节点有集群编号且关联节点无集群编号时, 所有相连节点统一为检索节点的集群编号
                elif point_group > 0 and min_no == 0:
                    for point in parent_list:
                        group[point] = point_group
                    for point in son_list:
                        group[point] = point_group
                
                ##检索节点与关联节点均有集群编号时, 所有相连节点统一为其中最小的集群编号
                elif point_group > 0 and min_no > 0:
                    min_group = min(point_group, min_no)
                    for point in parent_list:
                        group[point] = min_group
                    for point in son_list:
                        group[point] = min_group
                    for item in group.items():
                        if item[1] in connect_no or item[1] == point_group:
                            group[item[0]] = min_group
                        else:
                            continue
        
        ##集群检验
        group_kind = list(set(group.values()))
        if len(group_kind) == 1:
            return True
        else:
            group_item = dict()
            for item in group_kind:
                kind_item = list()
                for it in group.items():
                    if it[1] == item:
                        kind_item.append(it[0])
                    else:
                        continue
                group_item[str(item)] = kind_item
                
            ##输出集群字典
            return group_item
                
        
        
        
    def RING_DIST(self, father_dict):
        """判断结构中是否存在有向环结构."""
        
        ##获取节点名称
        cols = list(father_dict.keys())
        
        ##建立节点关联矩阵
        DAG_ARR = pd.DataFrame(index = cols, columns = cols)
        for key in father_dict.keys(): 
            value = father_dict[key] 
            for origin in value:
                DAG_ARR.loc[origin, key] = 1 
                
        ##获取结构信息
        new_DAG = DAG_ARR.copy()
        
        ##简化网络结构
        while new_DAG.dropna(axis = 0, how = 'all').dropna(axis = 1, how = 'all').shape != new_DAG.shape or len(new_DAG.keys()) != new_DAG.sum().sum(): 
            
            ##去除起始点与终止点
            if new_DAG.dropna(axis = 0, how = 'all').dropna(axis = 1, how = 'all').shape != new_DAG.shape:
                new_index = new_DAG.dropna(axis = 0, how = 'all').dropna(axis = 1, how = 'all').axes
                index1, index2 = list(new_index[0]), list(new_index[1])
                new_col = [col for col in index1 if col in index2]
                
                ##更新化简后的结构信息
                new_DAG = new_DAG.loc[new_col, new_col]
                
                ##当节点小于3时，不可能存在环装结构
                if len(new_DAG.keys()) < 3:
                    return True
                else:
                    pass
            
            ##去除长路径上的节点
            else:
                
                ##从可能的最小环开始检索
                for num in range(3, len(new_DAG.keys()) + 1):
                    break_flag = 0
                    for comb in itools.combinations(new_DAG.keys(), num):
                        long_arr = new_DAG.loc[comb, comb]
                        long_shape = long_arr.dropna(axis = 0, how = 'all').dropna(axis = 1, how = 'all').shape
                        long_line = long_arr.sum().sum()  
                        
                        ##当存在环时，输出环
                        if long_shape == long_arr.shape:
                            ring = dict()
                            for name in comb: 
                                st_name = long_arr[name][long_arr[name] == 1].index[0]
                                ring[name] = [st_name]
                            return ring
                        
                        ##当节点连线数不足以构成环时，跳过
                        elif long_line < num:
                            pass
                        
                        ##当节点连线数足以构成环时，选择最短路径
                        else:
                            for name in comb:
                                if long_arr[name].sum() == 1 and long_arr.loc[name].sum() == 1:
                                    st_name = long_arr[name][long_arr[name] == 1].index[0]
                                    new_DAG.loc[st_name, name] = 0
                                    end_name = long_arr.loc[name][long_arr.loc[name] == 1].index[0]
                                    new_DAG.loc[name, end_name] = 0
                                elif long_arr.loc[name].sum() == 1:
                                    end_name = long_arr.loc[name][long_arr.loc[name] == 1].index[0]
                                    new_DAG.loc[name, end_name] = 0
                                elif long_arr[name].sum() == 1:
                                    st_name = long_arr[name][long_arr[name] == 1].index[0]
                                    new_DAG.loc[st_name, name] = 0
                            new_DAG = new_DAG.where(new_DAG == 1)
                            break_flag = 1
                            break
                    if break_flag == 1:
                        break
                    else:
                        pass
                                
        ##当矩阵满秩且连线数量刚好等于行数时，这些节点成为一个闭环
        col_na = list(new_DAG.keys())
        new_ring = dict()
        for i in col_na:
            father = list(new_DAG[i][new_DAG[i] == 1].index)
            new_ring[i] = father
        
        ##判断这个环是一个大环还是几个小环
        ##当大环集群为1时，输出该环
        group = self.GROUP_DIST(new_ring)
        if group is True:
            return new_ring
        
        ##当有几个小环时，输出第一个环
        else:
            one_ring = dict()
            for i in group['1']:
                one_ring[i] = new_ring.get(i)
            return one_ring
        
        
        
        
    ##爬山法网络结构
    def HCNew(self, father_dict, lag, del_prob = 0.3, lamb = 0.05, score_function = 'BCPS'):
        """采用爬山法训练网络结构, 采用HCBest算法. 
           随机选择两个节点, 如果两个节点之间不存在连线, 则新增连线; 
           如果两个节点之间存在连线, 则以del_prob的概率删除连线, 以1-del_prob的概率反转连线方向.
           若新结构中存在有向环, 则删去有向环中能使网络分数提高的那条连线.
           若新结构中存在不同的节点集群, 则遍历不同集群中各节点的连线, 添加使得网络分数提升最多的那条连线."""
        
        
        ##获取节点名称并且生成任意两个节点的组合数列表
        point_num = len(self.col_name)
        comb_list = list(itools.permutations(self.col_name, 2))
        total_comb = len(comb_list)
        score_change = list()
        point_bcps = self.Score_T(father_dict = father_dict, lamb = lamb, score_function = score_function) 

        
        ##设置训练步长
        for i in tqdm(range(lag)): 
            new_father = copy.deepcopy(father_dict)
            new_bcps = copy.deepcopy(point_bcps)
            
            ##随机选取两个节点, 并随机生成0~1的概率数
            random_no = np.random.randint(0, total_comb)
            random_comb = comb_list[random_no]
            point1, point2 = random_comb[0], random_comb[1] 
            origin1, origin2 = new_father[point1], new_father[point2]
            aim1, aim2 = origin1.count(point2), origin2.count(point1)
            random_prob = np.random.random()  
            
            ##当随机选取的两个节点不存在连线时, 添加连线
            if aim1 + aim2 == 0:
                origin2.append(point1)
                new_bcps[point2] = self.Score_F(son_index = self.col_name.index(point2), parent_index = [self.col_name.index(i) for i in origin2], lamb = lamb, score_function = score_function)
                
            ##当生成的随机数小于del_prob时,删除连线
            elif random_prob < del_prob:
                if aim1 > 0:
                    origin1.remove(point2)
                    new_bcps[point1] = self.Score_F(son_index = self.col_name.index(point1), parent_index = [self.col_name.index(i) for i in origin1], lamb = lamb, score_function = score_function)
                else:
                    origin2.remove(point1)
                    new_bcps[point2] = self.Score_F(son_index = self.col_name.index(point2), parent_index = [self.col_name.index(i) for i in origin2], lamb = lamb, score_function = score_function)
                    
            ##当生成的随机数大于del_prob时,改变连线方向
            else:
                if aim1 > 0:
                    origin1.remove(point2)
                    origin2.append(point1)
                    new_bcps[point1] = self.Score_F(son_index = self.col_name.index(point1), parent_index = [self.col_name.index(i) for i in origin1], lamb = lamb, score_function = score_function)
                    new_bcps[point2] = self.Score_F(son_index = self.col_name.index(point2), parent_index = [self.col_name.index(i) for i in origin2], lamb = lamb, score_function = score_function)
                else:
                    origin2.remove(point1)
                    origin1.append(point2)
                    new_bcps[point1] = self.Score_F(son_index = self.col_name.index(point1), parent_index = [self.col_name.index(i) for i in origin1], lamb = lamb, score_function = score_function)
                    new_bcps[point2] = self.Score_F(son_index = self.col_name.index(point2), parent_index = [self.col_name.index(i) for i in origin2], lamb = lamb, score_function = score_function)
            
            ##DAG判别（是否是有向无环图）
            ##是否有不同集群
            GROUP = self.GROUP_DIST(new_father)
            
            ##是否存在环
            RING = self.RING_DIST(new_father)
            
            
            ##当生成图不是DAG时
            while (GROUP is not True) or (RING is not True):
                
                ##如果生成图有不同集群
                if GROUP is not True:
                    score_list = list()
                    group_num = len(GROUP.keys())
                    group_list = tuple(GROUP.values())
                    for comb in itools.product(*group_list):
                        for series in itools.permutations(comb, group_num):
                            connect_dict = copy.deepcopy(new_father)
                            connect_bcps = copy.deepcopy(new_bcps)
                            for no in range(group_num - 1):
                                connect_dict[series[no]].append(series[no + 1])
                                connect_bcps[series[no]] = self.Score_F(son_index = self.col_name.index(series[no]), parent_index = [self.col_name.index(i) for i in connect_dict[series[no]]], lamb = lamb, score_function = score_function)
                            score_list.append((connect_bcps, connect_dict))
                    new_score = sorted(score_list, key = lambda x : sum(x[0].values()), reverse = True)
                    new_father = copy.deepcopy(new_score[0][1])
                    new_bcps = copy.deepcopy(new_score[0][0])
                    
                ##如果生成图存在环结构
                else:
                    score_list = list()
                    for item in RING.items(): 
                        del_dict = copy.deepcopy(new_father)
                        del_bcps = copy.deepcopy(new_bcps)
                        del_dict[item[0]].remove(item[1][0])
                        del_bcps[item[0]] = self.Score_F(son_index = self.col_name.index(item[0]), parent_index = [self.col_name.index(i) for i in del_dict[item[0]]], lamb = lamb, score_function = score_function)
                        score_list.append((del_bcps, del_dict))
                    new_score = sorted(score_list, key = lambda x : sum(x[0].values()), reverse = True)
                    new_father = copy.deepcopy(new_score[0][1])
                    new_bcps = copy.deepcopy(new_score[0][0])
                
                ##判断新结构是否是DAG
                GROUP = self.GROUP_DIST(new_father) 
                RING = self.RING_DIST(new_father) 
                    
            ##结构替换
            origin_score = sum(point_bcps.values())
            new_score = sum(new_bcps.values())
            if new_score > origin_score:
                father_dict = copy.deepcopy(new_father)
                point_bcps = copy.deepcopy(new_bcps) 
            else:
                pass
            
            ##分数趋势记录
            score_change.append(origin_score)
        pd_score = pd.Series(score_change, index = range(1, lag + 1))
        
        ##输出最优结构
        self.dag = (father_dict, pd_score, score_change)
        return (father_dict, pd_score, score_change)
                
        
        
        
    ##节点概率表
    def point_prob(self, son_index, father_index, hash_dict):
        """根据最终的网络结构, 计算得出节点的贝叶斯表"""
        
        ##获取信息
        son_list = np.asarray(self.train_data[:, son_index])
        son_set = list(set(son_list))
        son_kind = len(son_set)
        father_num = len(father_index)
        total_index = father_index[:]
        total_index.insert(0, son_index)
        
        ##当没有父类时
        if father_num == 0:  
            
            ##计算似然概率
            for kind in son_set:  
                mem = (np.asarray(son_list) == kind).sum() 
                deno = self.train_data.shape[0]
                value = [mem, deno, (mem + 1)/(deno + son_kind)]
                
                ##hash加密输入
                key = '{}|{}|{}|{}'.format(self.col_name[son_index], [self.col_name[i] for i in father_index], int(kind), [])
                hash_dict[key] = value
        
        ##当存在父类时
        else:
            
            ##各个节点类别
            feature_kind = list()
            for i in total_index:
                feature = np.asarray(self.train_data[:, i])
                feature_set = tuple(set(feature))
                feature_kind.append(feature_set)
                
            ##各特征组合
            for comb in itools.product(*feature_kind):
                
                ##计算似然概率
                mem = (np.asarray(self.train_data[:, total_index]) == comb).min(1).sum()
                if father_num == 1:
                    deno = (np.asarray(self.train_data[:, father_index]) == comb[1:]).sum()
                else:
                    deno = (np.asarray(self.train_data[:, father_index]) == comb[1:]).min(1).sum()
                value = [mem, deno, (mem + 1)/(deno + son_kind)]
                
                ##hash加密输入
                key = '{}|{}|{}|{}'.format(self.col_name[son_index], [self.col_name[i] for i in father_index], int(comb[0]), [int(i) for i in comb[1:]])
                hash_dict[key] = value
                
            

            
            
    ##概率图生成
    def Prob_Graph(self, DAG_dict):
        """生成概率图.
           输出：
               Graph: 贝叶斯概率图
               DAG_dict: 概率图结构
               kind_dict: 各特征类型"""
        
        ##生成概率图
        Graph = dict()
        kind_dict = dict()
        for key in tqdm(self.col_name):
            son_index = self.col_name.index(key)
            father_list = DAG_dict[key]
            father_index = [self.col_name.index(i) for i in father_list]
            self.point_prob(son_index, father_index, Graph)
            son_set = tuple(set(np.asarray(self.train_data[:, son_index])))
            kind_dict[key] = son_set
        self.graph = (Graph, DAG_dict, kind_dict)
        return (Graph, DAG_dict, kind_dict) 
    
    
    
    
    ##运行程序
    def Run(self, lag, max_tree, del_prob = 0.3, lamb = 0.001, score_function = 'BCPS'):
        """运行整个程序
           log:训练步长
           max_tree:最大带权生成树方法
           lamb:正则项系数"""
        
        ##生成互信息矩阵
        print('|','正在生成互信息矩阵'.center(100, '='), '|')
        start = time.time()
        self.arr = self.MI_ARR()
        end = time.time()
        print('|','消耗时间为：{:.4f}秒'.format(end - start).center(100, '='), '|')
        print('|','生成完毕'.center(100, '='), '|', '\n'*5)
        
        ##生成提取权重
        print('|','正在提取权重'.center(100, '='), '|')
        start = time.time()
        self.w = self.W_OUT(self.arr)
        end = time.time()
        print('|','消耗时间为：{:.4f}秒'.format(end - start).center(100, '='), '|')
        print('|','提取完毕'.center(100, '='), '|', '\n'*5)
        
        ##生成最大带权生成树 
        print('|','正在生成最大带权生成树'.center(100, '='), '|')
        start = time.time()
        if max_tree == 'Prim':
            self.tree = self.Prim_tree(self.w)
        elif max_tree == 'Kruskal':
            self.tree = self.Kruskal_tree(self.w)
        end = time.time()
        print('|','消耗时间为：{:.4f}秒'.format(end - start).center(100, '='), '|')
        print('|','生成完毕'.center(100, '='), '|', '\n'*5)
        
        ##训练网络结构
        print('|','正在训练网络结构'.center(100, '='), '|')
        start = time.time()
        self.dag = self.HCNew(father_dict = self.tree, lag = lag, del_prob = del_prob, lamb = lamb, score_function = score_function)
        end = time.time()
        print('|','消耗时间为：{:.4f}秒'.format(end - start).center(100, '='), '|')
        print('|','训练完毕'.center(100, '='), '|', '\n'*5)
        
        ##生成概率图
        print('|','正在生成概率图'.center(100, '='), '|')
        start = time.time()
        self.graph = self.Prob_Graph(self.dag[0])
        end = time.time()
        print('|','消耗时间为：{:.4f}秒'.format(end - start).center(100, '='), '|')
        print('|','生成完毕'.center(100, '='), '|')
        
        
        
        
    ##概率图储存
    def Save(self, save_path):
        """将概率图储存为pickle二进制文件
           path为储存路径, 文件需为pkl文件"""
        
        ##读取概率图以及储存路径
        save_graph = self.graph
        save_file = open(save_path, 'wb')
        
        ##存放概率图
        pickle.dump(save_graph, save_file)
        save_file.close()
        
        
        
        
        
        
        
        
class Bayes_Predict():
    """用于预测贝叶斯网"""
    
    
    
    
    ##输入概率图
    def __init__(self, load_path, data):
        """加载概率图"""
        
        ##读取pkl文件
        load_file = open(load_path, 'rb')
        model = pickle.load(load_file)
        load_file.close()
        
        ##获取概率图信息
        self.Graph = model[0]
        self.DAG = model[1]
        self.point_kind = model[2] 
        self.data = data
        self.col_name = list(data.keys())
        
        
    
    
    ##节点概率
    def point_prob(self, colname, new_data):
        """计算节点的概率表"""
        
        ##计算Hash值
        father_list = self.DAG[colname]
        son_series = new_data[colname]
        father_series = new_data[father_list]
        new_data['hash'] = new_data[colname].map(str)
        for father in father_list:
            new_data['hash'] = new_data['hash'].map(str) + '|' + new_data[father].map(str)
        new_data['hash'] = new_data['hash'].apply(lambda x: [int(i) for i in x.split('|')])
        new_data['hash'] = new_data['hash'].apply(lambda x: '{}|{}|{}|{}'.format(colname, father_list, int(x[0]), [int(i) for i in x[1:]]))
        new_data['hash'] = new_data['hash'].apply(lambda x: self.Graph.get(x)[2])
        
        ##输出Hash序列
        return new_data['hash']
        
    
    
    
    ##预测
    def Prob(self, predict_point):
        """预测各类概率"""
        
        ##建立概率储存表
        total_prob = pd.DataFrame()
        
        ##生成类别列表
        kind_set = [self.point_kind[i] for i in predict_point]
        for kind in itools.product(*kind_set):
            new_pd = pd.DataFrame()
            
            ##建立新数据集
            new_data = self.data.copy()
            for i in range(len(predict_point)):
                new_data[predict_point[i]] = kind[i]
            
            ##计算概率
            col_list = list(new_data.keys())
            for col in col_list:
                hash_series = self.point_prob(col, new_data)
                new_pd[col] = hash_series
            total_prob[str(kind)] = new_pd.prod(axis = 1)
        
        ##标准化概率并输出预测结果
        sum_col = total_prob.sum(axis = 1) 
        df_prod = total_prob.div(sum_col, axis = 0) 
        self.pred = df_prod 
        return df_prod 
            
        
        
        
        
        
        
        
        
        
##应用
if __name__ == '__main__':
    
    ##读取数据
    data = pd.read_csv(r'E:\pyfile\melon2.csv')
    
    ##加载库
    bayes = Bayes_Net(data)
    
    ##训练
    bayes.Run(lag = 100, max_tree = 'Prim', lamb = 0.01, score_function = 'BIC')
    
    ##查看结构
    bayes.dag[0] #or by.graph[1]
    
    ##查看概率表
    bayes.graph[0]
    
    ##模型储存
    save_path = r'E:\pymodel\melon.pkl'
    bayes.Save(save_path)
    
    ##模型预测
    ##加载模型
    load_path = r'E:\pymodel\melon.pkl'
    pre_data = data[list(data.keys())[1:]]
    pre = Bayes_Predict(load_path, pre_data)
    
    ##预测
    pred_point = ['Y1']
    pre.Prob(pred_point)
    
    ##预测结果
    print(pre.pred)