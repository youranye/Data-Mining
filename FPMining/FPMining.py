#coding=utf-8                        # 全文utf-8编码
import sys
import time

def read(file):
    #读文件，转化为列表的列表
    dataset=[]
    fp=open(file,'r')
    for line in fp:
        line=line.strip('\n') #discard '\n'
        if line!='':
            dataset.append(line.split(','))
    fp.close()
    return dataset

#=========================================================================================================
# Apriori算法
#=========================================================================================================

def apriori(dataset:list, min_sup:int):
	
	'''L为候选k-项集的列表
 	C为候选k-项集的计数字典
	FI为频繁项集的列表
	'''
	start = time.time()
	C1={} 
	for T in dataset:  #产生候选1-项集的计数字典
		for I in T:
			if I in C1:
				C1[I] += 1
			else:
				C1[I] = 1
	L = []
	FI = []
	L1 = Apriori_prune(C1,min_sup)
	FI = FI + L1
	L = Apriori_gen(L1,len(L1)) #候选2-项集的列表
	k=2
	while L != []:
		C = dict()
		C = Apriori_count_subset(L,len(L),dataset) #C为候选k-项集的计数字典
		FI = FI + Apriori_prune(C,min_sup) 
		L = Apriori_gen(Apriori_prune(C,min_sup),len(Apriori_prune(C,min_sup)))
		k += 1
	end = time.time()
	return FI, end, start

def Apriori_gen(Itemset:list, length):
    """To generate new k-itemsets"""
    candidate = []
    for i in range (0,length):
        for j in range (i+1,length):
            if check_same_except_one(Itemset[i],Itemset[j]):
                candidate.append(list(set(Itemset[i]+Itemset[j]))) #Combine (k-1)-Itemset to k-Itemset 
    return candidate

def check_same_except_one(list1, list2):
    differing_count = len(set(list1)^set(list2))

    return differing_count == 2


def Apriori_prune(Ck, min_sup):
    '''剪枝步：支持度计数大于等于min_sup就加入这个候选项集'''
    L = []
    for i in Ck:
        if Ck[i] >= min_sup:
            L.append([j for j in i.split(',')])
    return L
 

def Apriori_count_subset(Candidate,Candidate_len,dataset):
    """ 对于候选k-项集列表，给出它的计数字典 """
    C = dict()
    for l in dataset:
        for i in range(0,Candidate_len):
            key = ','.join(j for j in Candidate[i])
            if key not in C: 
                C[key] = 0
            flag = True
            for k in Candidate[i]:
                if k not in l:
                    flag = False
            if flag:
                C[key] += 1
    return C

#===============================================================================================================
#FPTree-Growth算法
#===============================================================================================================

def FPGrowth(dataset:list,min_sup):
    initSet = create_initialset(dataset)
    start = time.time()
    FPtree, HeaderTable = create_FPTree(initSet, min_sup)
    frequent_itemset = []
    Mine_Tree(FPtree, HeaderTable, min_sup, set([]), frequent_itemset)
    end = time.time()
    frequent_itemset = [list(i) for i in frequent_itemset]
    return frequent_itemset, end, start

#To convert initial transaction into frozenset
def create_initialset(dataset):
    retDict = {}
    for trans in dataset:
        retDict[frozenset(trans)] = 1
    return retDict

#class of FP TREE node
class TreeNode:
    def __init__(self, Node_name,counter,parentNode):
        self.name = Node_name
        self.count = counter
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}
        
    def increment_counter(self, counter):
        self.count += counter

#To create Headertable and ordered itemsets for FP Tree
def create_FPTree(dataset, minSupport):
    HeaderTable = {}
    for transaction in dataset:
        for item in transaction:
            HeaderTable[item] = HeaderTable.get(item,0) + dataset[transaction]
    for k in list(HeaderTable):
        if HeaderTable[k] < minSupport:
            del(HeaderTable[k])

    frequent_itemset = set(HeaderTable.keys())

    if len(frequent_itemset) == 0:
        return None, None

    for k in HeaderTable:
        HeaderTable[k] = [HeaderTable[k], None]

    retTree = TreeNode('Null Set',1,None)
    for itemset,count in dataset.items():
        frequent_transaction = {}
        for item in itemset:
            if item in frequent_itemset:
                frequent_transaction[item] = HeaderTable[item][0]
        if len(frequent_transaction) > 0:
            #to get ordered itemsets form transactions
            ordered_itemset = [v[0] for v in sorted(frequent_transaction.items(), key=lambda p: p[1], reverse=True)]
            #to update the FPTree
            updateTree(ordered_itemset, retTree, HeaderTable, count)
    return retTree, HeaderTable

#To create the FP Tree using ordered itemsets
def updateTree(itemset, FPTree, HeaderTable, count):
    if itemset[0] in FPTree.children:
        FPTree.children[itemset[0]].increment_counter(count)
    else:
        FPTree.children[itemset[0]] = TreeNode(itemset[0], count, FPTree)

        if HeaderTable[itemset[0]][1] == None:
            HeaderTable[itemset[0]][1] = FPTree.children[itemset[0]]
        else:
            update_NodeLink(HeaderTable[itemset[0]][1], FPTree.children[itemset[0]])

    if len(itemset) > 1:
        updateTree(itemset[1::], FPTree.children[itemset[0]], HeaderTable, count)

#To update the link of node in FP Tree
def update_NodeLink(Test_Node, Target_Node):
    while (Test_Node.nodeLink != None):
        Test_Node = Test_Node.nodeLink

    Test_Node.nodeLink = Target_Node

#To transverse FPTree in upward direction
def FPTree_uptransveral(leaf_Node, prefixPath):
 if leaf_Node.parent != None:
    prefixPath.append(leaf_Node.name)
    FPTree_uptransveral(leaf_Node.parent, prefixPath)

#To find conditional Pattern Bases
def find_prefix_path(basePat, TreeNode):
 Conditional_patterns_base = {}

 while TreeNode != None:
    prefixPath = []
    FPTree_uptransveral(TreeNode, prefixPath)
    if len(prefixPath) > 1:
        Conditional_patterns_base[frozenset(prefixPath[1:])] = TreeNode.count
    TreeNode = TreeNode.nodeLink

 return Conditional_patterns_base

#function to mine recursively conditional patterns base and conditional FP tree
def Mine_Tree(FPTree, HeaderTable, minSupport, prefix, frequent_itemset):
    bigL = [v[0] for v in sorted(HeaderTable.items(),key=lambda p: p[1][0])]
    for basePat in bigL:
        new_frequentset = prefix.copy()
        new_frequentset.add(basePat)
        #add frequent itemset to final list of frequent itemsets
        frequent_itemset.append(new_frequentset)
        #get all conditional pattern bases for item or itemsets
        Conditional_pattern_bases = find_prefix_path(basePat, HeaderTable[basePat][1])
        #call FP Tree construction to make conditional FP Tree
        Conditional_FPTree, Conditional_header = create_FPTree(Conditional_pattern_bases,minSupport)

        if Conditional_header != None:
            Mine_Tree(Conditional_FPTree, Conditional_header, minSupport, new_frequentset, frequent_itemset)

#===============================================================================================================
#eclat算法
#===============================================================================================================
 
def eclat(dataset:list, min_sup):
    start = time.time()
    n_trans = len(dataset)
    '''生成初始频繁1项集的垂直表示'''
    f1_tid = {}
    f1_sup = {}
    c1_set = frozenset([item for transaction in dataset for item in transaction])
    for item in c1_set:
        c1_tid = set()
        support_count = 0
        for tid in range(n_trans):
            if item in dataset[tid]:
                c1_tid.add(tid)
                support_count += 1
        if support_count >= min_sup:
            itemset = frozenset([item])
            f1_tid[itemset] = frozenset(c1_tid)
            f1_sup[itemset] = support_count
    vertical_f_gen = time.time()

    sup_dict = gen_fk(f1_tid, f1_sup, min_sup)
    keys = sup_dict.keys()
    FI = []
    for i in keys:
        lst = []
        for j in i:
           lst.append(j) 
        FI.append(lst)
    end = time.time()
    return FI, end, start, vertical_f_gen

def gen_fk(f_tid, sup_dict, min_support_count):
    '''
    递归生成频繁项集
    :param f_tid: 频繁项集与tidset的字典，数据结构{f_set:f_set}
    :param sup_dict: 存储所有频繁项集的字典，键为频繁项集，值为支持度，数据结构{f_set:int}
    :param min_support_count: 最小支持度计数
    :return: 频繁项集的支持度计数字典
    '''
    f_set = list(f_tid.keys())
    tid_set = list(f_tid.values())
    n_item = len(f_set)

    for i in range(n_item-1):
        fp_tid = {}
        for j in range(i+1, n_item):
            if check_same_except_one(f_set[i],f_set[j]): #这个函数在apriori中写过
                cp_set = f_set[i] | f_set[j]
                cp_tid = tid_set[i] & tid_set[j]
                support_count = len(cp_tid)
                if support_count >= min_support_count:
                    fp_tid[cp_set] = cp_tid
                    sup_dict[cp_set] = support_count
        if len(fp_tid) > 1:
            gen_fk(fp_tid, sup_dict, min_support_count)
    return sup_dict


    
D = read('DBLPdata-10k.txt')
F_apriori, end_apriori, start_apriori = apriori(D, 5)
print('frequent itemset by Apriori:')
print(F_apriori)


F_fpgrowth, end_fpgrowth, start_fpgrowth = FPGrowth(D, 5)
print('frequent itemset by FP-Growth:')
print(F_fpgrowth)


F_eclat, end_eclat, start_eclat, vertical_f_gen = eclat(D, 5)
print('\nfrequent itemset by eclat:\n', F_eclat)


print("Time Taken by Apriori is:")
print(end_apriori-start_apriori)
print("Time Taken by FP-Growth is:")
print(end_fpgrowth-start_fpgrowth)
print("Time Taken by eclat is:")
print(end_eclat-start_eclat)
print("Time Taken by generating vertical data form is:")
print(vertical_f_gen-start_eclat)






