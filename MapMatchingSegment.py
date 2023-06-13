import os
import json
import pickle
import networkx as nx
from ConcentratedSubgraph import ConcentratedSubgraph

class MapMatchingSegment:
    def __init__(self) -> None:
        self.years_list = [y for y in range(1996, 2021)]
        self.path = "/data01/bruceData/tempfiles/TopicEvolutionVersion2/MapMatchingSegment"
       
       #加载图
        self.G = None
        self._load_graph()
        self.reachable_nodes_index = self._load_graph_reachable_index()
       
        #加载原始轨迹数据
        self.raw_trajectory = {}
        self._load_raw_trajetories()
       
        #加载原始轨迹和主题的关系的数据：通过年份和主题
        self.pid_to_topic = {}
        self._load_pid_to_topic()

    def _load_graph()-> None:
        graph_path = os.path.join("/data01/bruceData/tempfiles/TopicEvolutionVersion2/MapConstructorSegment", "Graph", 'map.gexf')
        self.G = nx.read_gexf(graph_path)

    def _load_graph_reachable_index(self) -> dict:
        '''
        使用字典（dictionary）构建一个双层索引来保存每个节点的连通节点。
        节点ID为第一层索引，年份为第二层索引，并保存每个节点ID在每个年份下的连通节点ID
        :return: {node:{year:{node set}}}
        '''

        reachable_nodes_dict = {}
        sorted_nodes = list(nx.topological_sort(self.G)) #对有向无环图的拓扑排序

        for node in sorted_nodes:
            reachable_nodes = set()
            reachable_nodes.add(node)

            for predecessor in self.G.predecessors(node): #此方法返回前驱节点
                predecessor_reachable_nodes = reachable_nodes_dict.get(predecessor, set())
                reachable_nodes.update(predecessor_reachable_nodes) # 状态转移方程： 当前节点的可达节点 = 自身 + 前驱节点的可达节点
            
            reachable_nodes_dict[node] = reachable_nodes
        
        #对reachable_nodes的每个node的值再按照year再分区
        reachable_nodes_index = {}
        for node, reachable_nodes in reachable_nodes_dict.items():
            node_reachable_nodes = {}
            for reachable_node in reachable_nodes:
                year, _ = reachable_node.split('_') 
                if year not in node_reachable_nodes:
                    node_reachable_nodes[year] = set()
                node_reachable_nodes[year].add(reachable_node)
            reachable_nodes_index[node] = node_reachable_nodes

        return reachable_nodes_index
    
    def _load_raw_trajetories(self)->None:
        '''
        从文件中读取原始轨迹
        '''
        raw_trajectories_path = os.path.join("/data01/bruceData/tempfiles/TopicEvolutionVersion2/RawTrajectoryGenerator", "raw_trajectory.jsonl")
        with open(os.path.join(raw_trajectories_path), 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.raw_trajectory.update(item)

    def _load_pid_to_topic(self):
        '''
        self.pid_to_topic
        用于将论文的pid映射为主题的编号的键：
            key: pid String
            value: (year, topic) Tuple, 第一个是对应的年份，第二个是该年份下的主题

        '''

        for year in self.years_list:
            labels_file = os.path.join("/data01/bruceData/tempfiles/TopicEvolutionVersion2/MapConstructorSegment", "cluster_lables", f'labels_{year}.pkl')
            with open(labels_file, 'rb') as f:
                labels, paper_ids = pickle.load(f)
            for label, pid in zip(labels, paper_ids):
                self.pid_to_topic[pid] = (int(year), int(label)) #pid: (year, label)

    def map_to_topic_trajectory(self) -> None:
        '''
        将原始轨迹映射为主题轨迹
        原始轨迹的基本格式（字典类型）： "pid1": [["pid2", 1998], ["pid3", 2000], ["pid1", 2001]]
        映射后的主题轨迹的基本格式："pid1": [("pid2", (1998, topic1)), ("pid3", (2000, topic2)), ("pid1", (2001, topic1))]

        '''
        topic_trajectories_path = os.path.join(self.path, "topic_trajectory.jsonl")
        if os.path.exists(topic_trajectories_path):
            print(f"topic_trajectories，文件已经存在，不再继续生成。\n 注意：如果演化图重新生成了，则这个文件必须重新生成！")
            return

        topic_trajectories = {}
        for pid, tra in self.raw_trajectory.items():
            # if year in self.years_list 后期可以去掉，暂时加上是防止获取到没有转换为主题的原始轨迹
            topic_trajectories[pid] = [(p_id, self.pid_to_topic[p_id]) for p_id, year in tra if year in self.years_list]

        # 保存原始主题轨迹
        self._save_topic_trajectory(topic_trajectories)

    def _generate_concentrated_subgraph(self, pid: str, one_topic_trajectory: list) -> list:
        '''
        :param one_topic_trajectory: pid, tra in topic_trajectories.items()
        topic_trajecories: {pid: [(pid, (1998, topic1)),...], ...}
        因此，one_topic_trajectory是一个元组列表
        :returns: 返回一个nodeid列表，nodeid是一个string，f"{year}_{topic}"        '''
        root_id = None
        root_year = None
        for item in reversed(one_topic_trajectory):#找到根节点
            if item[0] == pid:
                root_id = item
                root_year = item[1][0]
        
        root_id = (root_id[0], f"{root_id[1][0]}_{root_id[1][0]}") #(pid, year_topic)
        concentrated_graph = ConcentratedSubgraph(root_id,self.reachable_nodes_index)

        for new_node in reversed(one_topic_trajectory):
            if root_year > new_node[1][0]: #仅仅寻找年份之前的
                new_node = (new_node[0],f"{new_node[1][0]}_{new_node[1][1]}")
                concentrated_graph.add_node(new_node)

        res = concentrated_graph.get_concentrated_trajectories()
        map_tra_len = concentrated_graph.max_level + 1

        return res, map_tra_len


 
