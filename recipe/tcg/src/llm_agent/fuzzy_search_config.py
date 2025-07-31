# 模糊搜索配置文件
# 您可以通过修改这些参数来调整模糊搜索的行为

# 是否启用模糊搜索
# True: 启用模糊搜索，会尝试找到相似的已缓存查询
# False: 只进行精确匹配
FUZZY_SEARCH_ENABLED = True

# 相似度阈值 (0.0 - 1.0)
# 0.0: 完全不相似也会匹配 (不推荐)
# 0.5: 中等相似度匹配
# 0.7: 较高相似度匹配 (推荐)
# 0.9: 非常高相似度匹配
# 1.0: 完全相同才匹配 (等同于精确匹配)
FUZZY_SIMILARITY_THRESHOLD = 0.7

# 最大候选检查数量
# 为了避免性能问题，限制检查的缓存条目数量
# 较大的值会提高匹配准确性但可能影响性能
FUZZY_MAX_CANDIDATES = 50

# Redis配置
REDIS_HOST = '10.200.99.220'
REDIS_PORT = 32132
REDIS_DB = 0
REDIS_EXPIRE = 7 * 24 * 3600  # 7天

# 调试配置
DEBUG_GEOGPT_SEARCH = False
DEFAULT_VERBOSE = False

# 查询标准化配置
# 是否在比较时移除标点符号
REMOVE_PUNCTUATION_IN_COMPARISON = False

# 是否启用查询标准化 (转小写、去除多余空格等)
ENABLE_QUERY_NORMALIZATION = True

# 批量搜索配置
DEFAULT_MAX_WORKERS = 8
DEFAULT_TOPK = 2

# 模糊搜索匹配策略
# 'difflib': 使用Python内置的difflib.SequenceMatcher (默认)
# 'levenshtein': 使用编辑距离 (需要额外安装库)
# 'jaccard': 使用Jaccard相似度
SIMILARITY_METHOD = 'difflib'

def get_fuzzy_config():
    """
    获取当前的模糊搜索配置
    """
    return {
        'enabled': FUZZY_SEARCH_ENABLED,
        'threshold': FUZZY_SIMILARITY_THRESHOLD,
        'max_candidates': FUZZY_MAX_CANDIDATES,
        'similarity_method': SIMILARITY_METHOD,
        'normalize_query': ENABLE_QUERY_NORMALIZATION,
        'remove_punctuation': REMOVE_PUNCTUATION_IN_COMPARISON
    }

def update_fuzzy_config(**kwargs):
    """
    动态更新模糊搜索配置
    
    示例:
    update_fuzzy_config(threshold=0.8, max_candidates=100)
    """
    global FUZZY_SEARCH_ENABLED, FUZZY_SIMILARITY_THRESHOLD, FUZZY_MAX_CANDIDATES
    global SIMILARITY_METHOD, ENABLE_QUERY_NORMALIZATION, REMOVE_PUNCTUATION_IN_COMPARISON
    
    if 'enabled' in kwargs:
        FUZZY_SEARCH_ENABLED = kwargs['enabled']
    if 'threshold' in kwargs:
        if 0.0 <= kwargs['threshold'] <= 1.0:
            FUZZY_SIMILARITY_THRESHOLD = kwargs['threshold']
        else:
            raise ValueError("threshold must be between 0.0 and 1.0")
    if 'max_candidates' in kwargs:
        if kwargs['max_candidates'] > 0:
            FUZZY_MAX_CANDIDATES = kwargs['max_candidates']
        else:
            raise ValueError("max_candidates must be positive")
    if 'similarity_method' in kwargs:
        if kwargs['similarity_method'] in ['difflib', 'levenshtein', 'jaccard']:
            SIMILARITY_METHOD = kwargs['similarity_method']
        else:
            raise ValueError("similarity_method must be one of: difflib, levenshtein, jaccard")
    if 'normalize_query' in kwargs:
        ENABLE_QUERY_NORMALIZATION = kwargs['normalize_query']
    if 'remove_punctuation' in kwargs:
        REMOVE_PUNCTUATION_IN_COMPARISON = kwargs['remove_punctuation']

def print_fuzzy_config():
    """
    打印当前的模糊搜索配置
    """
    config = get_fuzzy_config()
    print("="*50)
    print("模糊搜索配置:")
    print(f"  启用状态: {config['enabled']}")
    print(f"  相似度阈值: {config['threshold']}")
    print(f"  最大候选数: {config['max_candidates']}")
    print(f"  相似度算法: {config['similarity_method']}")
    print(f"  查询标准化: {config['normalize_query']}")
    print(f"  移除标点符号: {config['remove_punctuation']}")
    print("="*50)
