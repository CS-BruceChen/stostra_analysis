from collections import defaultdict
import numpy as np
import json
import pickle

#直接对矩阵平方，会不会得到二级连接？

def second_segmenting() -> dict:
    '''
    读取轨迹数据，并且对轨迹数据进行分段，分段的含义是，找出连续年份对应的所有的轨迹片段
    
    :returns: 一个字典，字典的键是连续年份，例如"2001_2002"，字典的值是这两个年份中彼此之间有引用关系的论文id
    '''

    #raw_tarjectory.jsonl每一行是一个字典，字典有一个字段，key是轨迹id，value是轨迹列表，轨迹点[pid,year] {id: [[pid,year], ...], ..., id: [[pid, year], ...]}
    # raw_tra_file = os.path.join("/data01/bruceData/tempfiles/TopicEvolutionVersion2/RawTrajectoryGenerator", "raw_trajectory.jsonl")
    # if not os.path.exists(raw_tra_file):
    #     raise Exception
    
    raw_trajectory = {}
    with open('raw_trajectory.jsonl', 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            raw_trajectory.update(item) #添加字段到 raw_trajectory中

    second_segments_dict = defaultdict(list) #未定义的键返回默认列表
    for one_raw_tra in raw_trajectory.values():
        arranged_second_tra = defaultdict(list)
        for first_pid, _ in one_raw_tra:
            second_tra = raw_trajectory.get(first_pid,[])
            if second_tra:
                for second_pid, year in second_tra:
                    if year:
                        arranged_second_tra[year].append(second_pid)
        
        sorted_years = sorted(arranged_second_tra.keys()) #按年份排序，获取排序的年份列表

        for i in range(len(sorted_years) - 1):
            if sorted_years[i+1] == sorted_years[i] + 1: #年份相邻
                for pid1 in arranged_second_tra[sorted_years[i]]:
                    for pid2 in arranged_second_tra[sorted_years[i+1]]:
                        second_segments_dict[f"{sorted_years[i]}_{sorted_years[i+1]}"].append((pid1,pid2)) #连续年份对应的各自的论文的全连接

    return second_segments_dict



def second_compute_connection(segments_dict: dict, n_clusters: int=300) -> dict:
        '''
        根据片段计算连接性

        :param segments_dict: 存储轨迹片段的字典
        :returns: 归一化的连接性矩阵
        '''
        years_list = [y for y in range(1996, 2021)]


        general_connection_matrix = defaultdict(lambda: np.zeros((self.n_clusters,self.n_clusters))) #不存在的键默认创建n*n全0矩阵
        for i in range(len(years_list)-1):
            start_year, end_year = years_list[i],years_list[i+1] #滑动窗口遍历相邻年份
            start_year_pid_map_label = {} #pid映射到标签
            with open(f'labels_{start_year}.pkl', 'rb') as f:
                start_labels, start_paper_ids = pickle.load(f)  #将标签和论文ID一起保存
                for label, pid in zip(start_labels, start_paper_ids): #zip打包成元组列表，方便构建字典
                    start_year_pid_map_label[pid] = label
            end_year_pid_map_label = {}
            with open(f'labels_{end_year}.pkl', 'rb') as f:
                end_labels, end_paper_ids = pickle.load(f)  # 将标签和论文ID一起保存
                for label, pid in zip(end_labels, end_paper_ids):
                    end_year_pid_map_label[pid] = label
            
            #遍历segment
            pairs = segments_dict[f"{start_year}_{end_year}"]
            for start_pid, end_pid in pairs:
                start_label = start_year_pid_map_label[start_pid]
                end_label = end_year_pid_map_label[end_pid]
                general_connection_matrix[f"{start_year}_{end_year}"][start_label, end_label] += 1 #对连接次数进行统计，生成权重矩阵

        with open("general_connection_matrix.pkl", "wb") as f: #存储中间结果
            pickle.dump(dict(general_connection_matrix), f)

        weighted_normalized_connection_matrix = defaultdict(lambda: np.zeros((self.n_clusters, self.n_clusters))) #归一化连接矩阵
        for i in range(len(years_list) - 1):
            start_year, end_year = years_list[i], years_list[i+1]
            start_year_labels_map_count = defaultdict(int)
            with open(f'labels_{start_year}.pkl', 'rb') as f:
                start_labels, _ = pickle.load(f)  # 将标签和论文ID一起保存
                for label in start_labels:
                    start_year_labels_map_count[label] += 1 #对每个标签进行计数

            end_year_labels_map_count = defaultdict(int)            
            with open(f'labels_{end_year}.pkl', 'rb') as f:
                end_labels, _ = pickle.load(f)  # 将标签和论文ID一起保存
                for label in end_labels:
                    end_year_labels_map_count[label] += 1        

            for start_label in range(n_clusters):
                for end_label in range(n_clusters): #修改：有可能可以矩阵加速
                    origin = general_connection_matrix[f"{start_year}_{end_year}"][start_label, end_label]
                    total = start_year_labels_map_count[start_label] * end_year_labels_map_count[end_label]
                    weighted_normalized_connection_matrix[f"{start_year}_{end_year}"][start_label, end_label] = origin / total #修改：如此定义层次之间的聚类连接性，但是依据是什么？
            
            min_val = np.min(weighted_normalized_connection_matrix[f"{start_year}_{end_year}"], axis=1, keepdims=True)  # 计算每行的最小值
            max_val = np.max(weighted_normalized_connection_matrix[f"{start_year}_{end_year}"], axis=1, keepdims=True)  # 计算每行的最大值
            weighted_normalized_connection_matrix[f"{start_year}_{end_year}"] = (weighted_normalized_connection_matrix[f"{start_year}_{end_year}"] - min_val) / (max_val - min_val)

        with open("weighted_normalized_connection_matrix.pkl", "wb") as f:
            pickle.dump(dict(weighted_normalized_connection_matrix), f)
            print("加权归一化后的矩阵连接存储完成")

        return weighted_normalized_connection_matrix


if __name__ == '__main__':
    second_segments_dict = second_segmenting()
    second_compute_connection(second_segments_dict)
    pass