class DataLoader:
    '''
    数据加载器类，从MongoDB中加载数据

    MongoDB的文档数据结构如下
    id: 文档的 ID。这是数据中的一个字段，不是 MongoDB 自动生成的 _id 字段。
    title: 文章的标题。
    abstract: 文章的摘要。
    year: 文章的发布年份。
    
    '''

    def __init__(self, connection_string: str='xxxxx') -> None:
        '''
        初始化DataLoader，连接MongoDB，读取paper关系，并加载所有数据

        :param connection_string: MongoDB的主机地址字符串
        :type connection_string: str
        
        '''
        self.client = MogoClient(connection_string)
        self.db = self.client.AminerCitation
        self.paper = self.db.papers
        self._load_all_data()

    def _load_all_data(self) -> None:
       '''
       在db中查找所有年份在1996之后且摘要不为空的论文，并按照论文年份为键建立字典
       
       '''    
       
       projection = {"id": 1, "title": 1, "abstract": 1, "year": 1, "_id": 0}
       all_data = self.papers.find({"abstract": {"$ne": None, "$ne": ""}, "year": {"$gte": 1996}}, projection) #在mongodb中做查询：第一个参数指定查询条件，第二个参数指定返回字段，1表示返回，0表示不返回
       self.data_cache = {} #从命名角度更应该叫做paper_cache
       for document in all_data: #按年份建立字典，对返回的数据分类
           year = document.get('year')
           if year not in self.data_cache:
               self.data_cache[year] = []
            self.data_cache[year].append(document)
           
    def get_data_by_one_year(self, year: int) -> list:
        '''
        从data_cache字典返回特定年份的论文列表
        
        :param year: 年份
        :returns:
        [
            {
            "id": "paper_001",
            "title": "Example Paper Title",
            "abstract": "This is an example abstract.",
            "year": 2020
            },
            {
            "id": "paper_002",
            "title": "Another Example Paper Title",
            "abstract": "This is another example abstract.",
            "year": 2020
            },
            ...
        ]

        '''
        #year不存在返回空列表
        return self.data_cache.get(year, [])

    def get_data_by_range_year(self, start_year: int, end_year: int) -> list:
        '''
        返回data_cache字典中，在指定年份范围内的论文列表

        返回data_cache中，年份在start_year到end_year之间的论文，此处是闭区间

        :param start_year: 年份下界
        :param end_year: 年份上界
        :returns: 指定年份范围内的论文的列表

        '''

        return [document for year in range(start_year, end_year + 1) for document in self.data_cache.get(year, [])]
    
    def get_data_count_by_year(self, year: int) -> int:
        '''
        返回data_cache字典中特定年份的数据数目
        
        :param year: 年份
        :returns: 对应年份包含的论文的数目

        '''
        
        return len(self.data_cache.get(year, []))