import json
from pymongo import MongoClient
import pickle
import os

class RawTrajectoryGenerator:
    def __init__(self, mongo_url:str='mongodb://root:865067519@localhost:27017') -> None:
        '''
        连接数据库并读取paper数据
        然后读取paper聚类
        '''
        self.path = r"/data01/bruceData/tempfiles/TopicEvolutionVersion2/RawTrajectoryGenerator"
        self.client = MongoClient(mongo_url)
        self.db = self.client.AminerCitation
        self.papers = self.db.papers

        with open(os.path.join("/data01/bruceData/tempfiles/TopicEvolutionVersion2/MapConstructor", "cluster_paper_id_map_and_cluster_centroid_map.pkl"), 'rb') as f:
            self.cluster_paper_id_map, _ = pickle.load(f)
        self.raw_trajectory = {}

    def get_raw_trajectory(self) -> None:
        '''
        给出论文的所有引用
        字典，键是paper_id
        '''    
        for year, paper_ids in self.cluster_paper_id_map:
            for ii, paper_id in enumerate(paper_ids.keys()): #转换为枚举对象，从而可以输出下标
                paper_info = list(self.papers.find({"id": paper_id, "references": {"$exists": True, "$ne": []}}, {"id": 1, "references": 1, "_id": 0})) #查找非空引用并返回
                if paper_info:
                    refs_info = list(self.papers.find({"id": {"$in": paper_info[0]['references']}, "abstract": {"$ne": None, "$ne": ""}}, {"id": 1, "year": 1, "_id": 0})) #引用中摘要非空
                    refs_info = {info["id"]: info["year"] for info in refs_info if info["year"] >= 1996} #只保留1996年之后的文献
                    trajectory = [(paper_id, year)] + [(ref_id, refs_info[ref_id]) for ref_id in refs_info] #修改：这个地方可能有问题，因为ref_id是字典的字段，不能作为list的下标，并且，也只会拼接成元组列表
                    if len(trajectory) >= 3:
                        self.raw_trajectory[paper_id] = sorted(trajectory, key=lambda x: x[1])  # 修改：按年份排序,x[1]作为key有误，元组列表的第一个元素的第二项才是year，lambda x: x[0][1]
    
    def save_trajectory(self) -> None:
        with open(os.path.join(self.path, "raw_trajectory.jsonl"), 'w') as f:
            for key, value in self.raw_trajectory.items():
                f.write(json.dumps({key: value}) + '\n')

    def load_trajectory(self) -> dict:
        raw_trajectory = {}
        with open(os.path.join(self.path, "raw_trajectory.jsonl"), 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                raw_trajectory.update(item)
        return raw_trajectory