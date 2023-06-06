import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import os
import pickle
import json
from sklearn.decomposition import PCA
from DataLoader import DataLoader
from sklearn.cluster import KMeans
from collections import defaultdict
import networkx as nx

class MapConstructorSegment:
    def __init__(self, data_loader: DataLoader, n_clusters: int, embedded_name: str='all-MiniLM-L6-V2',
                 reduction_name: str="PCA", reduction_components: int=6, connection_threshold: float=0.8) -> None:
        '''初始化构造器

        指定数据源和模型以及各种参数的取值

        :param data_loader: 指定论文数据源
        :param n_clusters: 聚类个数
        :param embedded_name: 嵌入模型名称。默认为all-MiniLM-L6-V2
        :param reduction_name: 降维模型名称。默认为PCA
        :param reduction_components: 降维维度。默认为6
        :param connection_threshold: 连接阈值。默认为0.8
        '''
        
        self.path = r"/data01/bruceData/tempfiles/TopicEvolutionVersion2/MapConstructorSegment" #存放轨迹数据的目录
        self.reduction_components = reduction_components
        self.n_clusters = n_clusters
        self.connection_threshold = connection_threshold
        self.years_list = [y for y in range(1996, 2021)]
        self.data_loader = data_loader
        self.embedded_model, self.dimensionality_reduction_model = self._get_embedding_reduction_model(embedded_name, reduction_name)

    def _get_embedding_reduction_model(self, embedded_name: str, reduction_name: str) -> tuple:
        '''
        获取嵌入模型和降维模型

        :param embedded_name: 嵌入模型的名称
        :param reduction_name: 降维模型的名称
        :returns: 嵌入模型和降维模型
        '''
        
        embedded_model = None
        dimensionality_reduction_model = None
        if embedded_name == "all-MiniLM-L6-V2" and reduction_name == "PCA":
            embedded_model = SentenceTransformer(embedded_name, device="cuda:1" if torch.cuda.is_available() else "cpu")
            dimensionality_reduction_model = PCA(n_components=self.reduction_components)
        else:
            pass
        return embedded_model, dimensionality_reduction_model
    
    def construct(self) -> None:
        '''
        构建演化图

        6个步骤构建演化图，嵌入数据，降维数据，对数据聚类，读取原始轨迹，计算连接，最后从加权矩阵构建一个图
        '''
        
        self._embedding()
        self._reduction()
        self._clustering()
        segments_dict = self._segmenting()
        weighted_normalized_connection_matrix = self._compute_connection(segments_dict)
        self._construct_graph(weighted_normalized_connection_matrix)

    def _embedding(self) -> None:
        '''
        读取每一年的论文题目和摘要，用嵌入模型编码，和论文id一起写入对应年份的pkl文件保存起来
        
        '''
        for year in self.years_list:
            embedding_file = os.path.join(self.path,"full_embeddings",f'embeddings_{year}.pkl')
            if os.path.exists(embedding_file):
                continue #避免重复新建文件

            papers = self.data_loader.get_data_by_one_year(year)
            title_and_abstracts = [paper['title'] + "." + paper['abstract'] for paper in papers]
            paper_ids = [paper['id'] for paper in papers]
            embeddings = self.embedded_model.encode(title_and_abstracts, batch_size=2048)

            with open(embedding_file,'wb') as f:
                pickle.dump((embeddings, paper_ids), f)

    def _reduction(self) -> None:
        '''
        读取嵌入数据，使用降维模型处理，并保存得到降维后的嵌入数据
        
        '''
        for year in self.years_list:
            reduction_embedding_file = os.path.join(self.path, "reduction_embedding", f'reduction_embeddings_{year}.pkl')
            if os.path.exists(reduction_embedding_file):
                continue #避免重复新建文件
        
            full_embedding_file = os.path.join(self.path, "full_embeddings", f'embeddings_{year}.pkl') #读取嵌入文件
            with open(full_embedding_file, 'rb') as f:
                full_embedding, paper_ids = pickle.load(f)
            
            reduction_embeddings = self.dimensionality_reduction_model.fit_transform(full_embedding) #用降维模型降维

            #第三步，将降维后的嵌入保存
            with open(reduction_embedding_file, 'wb') as f:
                pickle.dump((reduction_embeddings, paper_ids), f)  # 将嵌入和论文ID一起保存


    def _clustering(self) -> None:
        '''
        读取降维后的数据，使用Kmeans函数聚类，然后保存标签文件
        '''
        for year in self.years_list:
            labels_file = os.path.join(self.path, "cluster_lables", f'labels_{year}.pkl')
            if os.path.exists(labels_file):
                continue #避免重复新建文件

            reduction_embedding_file = os.path.join(self.path, "reduction_embedding", f'reduction_embeddings_{year}.pkl')
            with open(reduction_embedding_file, "rb") as f:
                reduction_embeddings, paper_ids = pickle.load(f)

            kmeans = KMeans(n_clusters=self.n_clusters)
            labels = kmeans.fit_predict(reduction_embeddings)

            with open(labels_file, 'wb') as f:
                pickle.dump((labels, paper_ids), f)  # 将标签和论文ID一起保存

    def _segmenting(self) -> dict:
        '''
        读取轨迹数据，并且对轨迹数据进行分段，分段的含义是，找出连续年份对应的所有的轨迹片段
        
        :returns: 一个字典，字典的键是连续年份，例如"2001_2002"，字典的值是这两个年份中彼此之间有引用关系的论文id
        '''

        raw_tra_file = os.path.join("/data01/bruceData/tempfiles/TopicEvolutionVersion2/RawTrajectoryGenerator", "raw_trajectory.jsonl")
        if not os.path.exists(raw_tra_file):
            raise Exception
        
        #raw_tarjectory.jsonl每一行是一个字典，字典有一个字段，key是轨迹id，value是轨迹列表，轨迹点[pid,year]

        raw_trajectory = {}
        with open(raw_tra_file, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                raw_trajectory.update(item) #添加字段到 raw_trajectory中

        segments_dict = defaultdict(list) #未定义的键返回默认列表
        for one_raw_tra in raw_trajectory.values():
            arranged_tra = defaultdict(list)
            for pid, year in one_raw_tra:
                arranged_tra[year].append(pid) #按照年份对轨迹点进行分类
            
            sorted_years = sorted(arranged_tra.keys()) #按年份排序，获取排序的年份列表

            for i in range(len(sorted_years) - 1):
                if sorted_years[i+1] == sorted_years[i] + 1: #年份相邻
                    for pid1 in arranged_tra[sorted_years[i]]:
                        for pid2 in arranged_tra[sorted_years[i+1]]:
                            segments_dict[f"{sorted_years[i]}_{sorted_years[i+1]}"].append((pid1,pid2)) #连续年份对应的各自的论文的全连接

        return segments_dict
    
    def _compute_connection(self, segments_dict: dict) -> dict:
        '''
        根据片段计算连接性

        :param segments_dict: 存储轨迹片段的字典
        :returns: 归一化的连接性矩阵
        '''
        general_connection_matrix = defaultdict(lambda: np.zeros((self.n_clusters,self.n_clusters))) #不存在的键默认创建n*n全0矩阵
        for i in range(len(self.years_list)-1):
            start_year, end_year = self.years_list[i],self.years_list[i+1] #滑动窗口遍历相邻年份
            start_year_pid_map_label = {} #pid映射到标签
            with open(os.path.join(self.path, "cluster_lables", f'labels_{start_year}.pkl'), 'rb') as f:
                start_labels, start_paper_ids = pickle.load(f)  #将标签和论文ID一起保存
                for label, pid in zip(start_labels, start_paper_ids): #zip打包成元组列表，方便构建字典
                    start_year_pid_map_label[pid] = label
            end_year_pid_map_label = {}
            with open(os.path.join(self.path, "cluster_lables", f'labels_{end_year}.pkl'), 'rb') as f:
                end_labels, end_paper_ids = pickle.load(f)  # 将标签和论文ID一起保存
                for label, pid in zip(end_labels, end_paper_ids):
                    end_year_pid_map_label[pid] = label
            
            #遍历segment
            pairs = segments_dict[f"{start_year}_{end_year}"]
            for start_pid, end_pid in pairs:
                start_label = start_year_pid_map_label[start_pid]
                end_label = end_year_pid_map_label[end_pid]
                general_connection_matrix[f"{start_year}_{end_year}"][start_label, end_label] += 1 #对连接次数进行统计，生成权重矩阵

        with open(os.path.join(self.path, "Graph", "general_connection_matrix.pkl"), "wb") as f: #存储中间结果
            pickle.dump(dict(general_connection_matrix), f)

        weighted_normalized_connection_matrix = defaultdict(lambda: np.zeros((self.n_clusters, self.n_clusters))) #归一化连接矩阵
        for i in range(len(self.years_list) - 1):
            start_year, end_year = self.years_list[i], self.years_list[i+1]
            start_year_labels_map_count = defaultdict(int)
            with open(os.path.join(self.path, "cluster_lables", f'labels_{start_year}.pkl'), 'rb') as f:
                start_labels, _ = pickle.load(f)  # 将标签和论文ID一起保存
                for label in start_labels:
                    start_year_labels_map_count[label] += 1 #对每个标签进行计数

            end_year_labels_map_count = defaultdict(int)            
            with open(os.path.join(self.path, "cluster_lables", f'labels_{end_year}.pkl'), 'rb') as f:
                end_labels, _ = pickle.load(f)  # 将标签和论文ID一起保存
                for label in end_labels:
                    end_year_labels_map_count[label] += 1        

            for start_label in range(self.n_clusters):
                for end_label in range(self.n_clusters): #修改：有可能可以矩阵加速
                    origin = general_connection_matrix[f"{start_year}_{end_year}"][start_label, end_label]
                    total = start_year_labels_map_count[start_label] * end_year_labels_map_count[end_label]
                    weighted_normalized_connection_matrix[f"{start_year}_{end_year}"][start_label, end_label] = origin / total #修改：如此定义层次之间的聚类连接性，但是依据是什么？
            
            min_val = np.min(weighted_normalized_connection_matrix[f"{start_year}_{end_year}"], axis=1, keepdims=True)  # 计算每行的最小值
            max_val = np.max(weighted_normalized_connection_matrix[f"{start_year}_{end_year}"], axis=1, keepdims=True)  # 计算每行的最大值
            weighted_normalized_connection_matrix[f"{start_year}_{end_year}"] = (weighted_normalized_connection_matrix[f"{start_year}_{end_year}"] - min_val) / (max_val - min_val)

        with open(os.path.join(self.path, "Graph", "weighted_normalized_connection_matrix.pkl"), "wb") as f:
            pickle.dump(dict(weighted_normalized_connection_matrix), f)
            print("加权归一化后的矩阵连接存储完成")

        return weighted_normalized_connection_matrix
        
    def _construct_graph(self, weighted_normalized_connection_matrix: dict) -> None:
        '''
        构造演化图，不返回任何值，但是把演化图存储为gexf文件

        :param: 归一化的连接性矩阵
        
        '''

        bool_connection_matrix = defaultdict(lambda: np.zeros_like(weighted_normalized_connection_matrix, dtype=bool)) #zeros_like创建一个形状相似的全0矩阵
        for year_year, arr in weighted_normalized_connection_matrix.items():
            meanVal = np.mean(arr, axis=1, keepdims=True)
            bool_connection_matrix[year_year] = (arr >= meanVal) # 每一行大于等于平均值的位置置位True，否则为False，也就是大于均值才认为是有连接的

        with open(os.path.join(self.path, "Graph", "bool_connection_matrix.pkl"), "wb") as f:
            pickle.dump(dict(bool_connection_matrix), f)

        self._connection_matrix_to_map(bool_connection_matrix) #根据连接性图生成演化图

    def _connection_matrix_to_map(self, bool_connection_matrix):
        '''
        把邻接矩阵转换为图
        '''
        G = nx.DiGraph() #空的有向图
        for year_year, arr in bool_connection_matrix.items():
            # 获取开始的年份，连续一年的年份
            start_year, end_year = year_year.split("_")
            for start_topic in range(arr.shape[0]): #numpy shape获取维度信息
                start_node_id = f"{start_year}_{start_topic}" #topic是以自然序列编码的
                G.add_node(start_node_id) #添加节点
                for end_topic in range(arr.shape[1]):
                    end_node_id = f"{end_year}_{end_topic}"
                    G.add_node(end_node_id) #添加节点

                    #添加边
                    if arr[start_topic, end_topic] == True:
                        G.add_edge(start_node_id, end_node_id)            


        # 将图保存为通用的文件格式
        nx.write_gexf(G, os.path.join(self.path, "Graph", 'map.gexf'))