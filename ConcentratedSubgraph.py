from collections import defaultdict

class ConcentratedSubgraph:
    def __init__(self, root_node_id, reachable_nodes_index) -> None:
        '''
        :param: root_node_id: (pid,"{year}_{topic}")
        :param: reachable_nodes_index: {node:{year:{node set}}}
        '''

        self.reachable_nodes_index = reachable_nodes_index
        self.subgraph = defaultdict(list) #节点和节点的前驱列表
        root_node = (root_node_id , 0) #(nodeid, level)
        self.subgraph[root_node] = [] #根节点的前驱列表为空
        self.max_level = 0 #子图中的最高层级
        self.nodes_by_level = defaultdict(list) #每个层级的节点列表
        self.nodes_by_level[0].append(root_node)

    def add_node(self, node_id):
        '''
        检查节点能否被索引到，并且对符合条件的节点，新增一个层次存储其中
        '''
        year, _ = node_id[1].split('_')
        for level in range(self.max_level, -1, -1):
            predecessors = []
            for node in self.nodes_by_level[level]:
                index_node_id = node[0][1] #node[0][1]:pid
                #条件：年份要满足 且 节点也要在对应年份的集合中
                if year in self.reachable_nodes_indexp[index_node_id] and node_id[1] in self.reachable_nodes_index[index_node_id][year]:
                    predecessors.append(node)

            if predecessors:
                new_node = (node_id, level + 1)
                self.subgraph[new_node] = predecessors
                self.max_level = max(self.max_level, level + 1)
                # 将新节点添加到相应层级的节点列表中。
                self.nodes_by_level[level + 1].append(new_node)
                return
        # 如果没有可以达到给定节点的节点，就返回。
        return                

    def get_concentrated_trajectories(self):
        concentrated_tra = [] #是一个node_id的列表
        stack = [(node,[node[0]]) for node in self.nodes_by_level[self.max_level]]
        

        while stack:
            node, path = stack.pop()
            if node[1] == 0: #node level is 0, 表明到达根节点了
                concentrated_tra.append(path)
            else:
                stack.extend((predecessor,[predecessor[0]] + path) for predecessor in self.subgraph[node])

        return concentrated_tra

    