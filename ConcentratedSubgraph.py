from collections import defaultdict

class ConcentratedSubgraph:
    def __init__(self, root_node_id: tuple, reachable_nodes_index: dict) -> None:
        '''
        构造函数，用于初始化浓缩子图。
        :param root_node_id: 根节点的ID，也就是子图中的核心节点。(pid,topic_id)
        :param reachable_nodes_index: 每个节点可以达到的节点的索引，这是一个双层字典，第一层的键是节点ID，第二层的键是年份，值是在该年份可以达到的节点的集合。
        '''
        self.reachable_nodes_index = reachable_nodes_index
        self.subgraph = defaultdict(list) #子图成员，键是节点，值是节点的前驱节点列表
        root_node =(root_node_id, 0) #(id,level)
        self.subgraph[root_node] = []

        self.max_level = 0
        self.nodes_by_level = defaultdict(list)
        self.nodes_by_level[0].append(root_node)

    def add_node(self, node_id: tuple) -> None:
        '''
        添加节点
        检验要添加的节点的合法性，然后搜寻其前驱节点，组织成节点的合法形式，加入子图并修改相关字典

        :param node_id: 要添加的节点(pid,topic_id),topic_id 形如 2021_topic1
        
        '''
        year, _ = node_id[1].split('_')
        #从最高已知层级开始搜索，然后逐级降低，直到找到可以到达给定节点的节点或者搜索到根节点对应层级
        for level in range(self.max_level, -1, -1): #从max_level到0的递减序列
            predecessors = []
            for node in self.nodes_by_level[level]:
                index_node_id = node[0][1] #node (("pid2", "2021_topic1"), 0),node[0][1]即为节点id，用于在reachable_nodes_index中进行索引
                #检查年份是否符合要求，以及年份对应的可达前驱节点中是否有node_id对应的节点
                if year in self.reachable_nodes_index[index_node_id] and node_id[1] in self.reachable_nodes_index[index_node_id][year]:
                    predecessors.append(node) #有则添加到列表作为node_id对应节点的前驱节点
            if predecessors:
                new_node = (node_id, level + 1)
                self.subgraph[new_node] = predecessors
                self.max_level = max(self.max_level,level + 1)
                self.nodes_by_level[level + 1].append(new_node)
                return
        return
    
    def get_concentrated_trajectories(self) -> list:
        '''
        使用构建好的图返回一个路径
        :returns: 节点列表用于表示路径
        '''

        paths = []
        stack = [(node,[node[0]]) for node in self.nodes_by_level[self.max_level]] #从最下层开始

        while stack: #深度优先搜索
            node, path = stack.pop()
            if node[1] == 0: #到达根节点所处层级
                paths.append(path) #返回当前路径作为结果
            else: #subgraph[node] 是node对应的前驱节点的列表，predecessor是单个前驱节点((pid,topic_id),level)
                stack.extend((predecessor, [predecessor[0]] + path) for predecessor in self.subgraph[node])
        
        return paths
    

    