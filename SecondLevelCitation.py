from collections import defaultdict
import numpy as np
from MapConstructorSegment import MapConstructorSegment
import json
import os

#直接对矩阵平方，会不会得到二级连接？

def second_segmenting(self) -> dict:
    '''
    读取轨迹数据，并且对轨迹数据进行分段，分段的含义是，找出连续年份对应的所有的轨迹片段
    
    :returns: 一个字典，字典的键是连续年份，例如"2001_2002"，字典的值是这两个年份中彼此之间有引用关系的论文id
    '''

    #raw_tarjectory.jsonl每一行是一个字典，字典有一个字段，key是轨迹id，value是轨迹列表，轨迹点[pid,year] {id: [[pid,year], ...], ..., id: [[pid, year], ...]}
    raw_tra_file = os.path.join("/data01/bruceData/tempfiles/TopicEvolutionVersion2/RawTrajectoryGenerator", "raw_trajectory.jsonl")
    if not os.path.exists(raw_tra_file):
        raise Exception
    
    raw_trajectory = {}
    with open(raw_tra_file, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            raw_trajectory.update(item) #添加字段到 raw_trajectory中

    second_segments_dict = defaultdict(list) #未定义的键返回默认列表
    for one_raw_tra in raw_trajectory.values():
        arranged_second_tra = defaultdict(list)
        for first_pid, _ in one_raw_tra:
            for second_pid, year in raw_trajectory[first_pid]:
                arranged_second_tra[year].append(second_pid)
        
        sorted_years = sorted(arranged_second_tra.keys()) #按年份排序，获取排序的年份列表

        for i in range(len(sorted_years) - 1):
            if sorted_years[i+1] == sorted_years[i] + 1: #年份相邻
                for pid1 in arranged_second_tra[sorted_years[i]]:
                    for pid2 in arranged_second_tra[sorted_years[i+1]]:
                        second_segments_dict[f"{sorted_years[i]}_{sorted_years[i+1]}"].append((pid1,pid2)) #连续年份对应的各自的论文的全连接

        return second_segments_dict

    


            


def compute_second_connection(mcs: MapConstructorSegment, segments_dict: dict) -> dict:
    '''
    :param segments_dict: {'2001-2002': (pid1, pid2), ...}
    '''
    general_connection_matrix = defaultdict(lambda: np.zeros((mcs.n_clusters, mcs.n_clusters))) #每个相邻年份的值是一个邻接矩阵
    for i in range(len(mcs.years_list)-1):    
        start_year, end_year = mcs.years_list[i], mcs.years_list[i+1] #滑动窗口遍历相邻年份
        


if __name__ == '__main__':
    print('SecondLevelCitation')
    pass