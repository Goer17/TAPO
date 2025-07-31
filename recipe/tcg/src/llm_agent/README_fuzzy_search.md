# GeoGPT 模糊搜索功能说明

## 功能概述

这个模糊搜索功能允许您在Redis缓存中进行智能的相似查询匹配，同时保持原始问题-答案对的完整存储。

## 主要特性

### 1. 双重缓存机制
- **精确匹配**: 首先尝试找到完全相同的查询
- **模糊匹配**: 如果没有精确匹配，则查找相似的查询

### 2. 原始存储保持
- 问题和答案在Redis中以原始形式存储
- 不会修改或压缩原始内容
- 保持数据的完整性和可读性

### 3. 智能相似度计算
- 使用Python内置的`difflib.SequenceMatcher`
- 支持中英文混合查询
- 可配置的相似度阈值

## 使用方法

### 基本使用
```python
from geogpt_topkcommon import geogpt_search, geogpt_batch_search

# 单个查询搜索
result, status = geogpt_search("中国的首都是什么？", topk=2, verbose=True)
print(f"结果: {result}")
print(f"状态: {status}")  # 可能是: 'exact_hit', 'fuzzy_hit', 'new', 'not_found'

# 批量查询搜索
queries = [
    "中国的首都是什么？",
    "What is the capital of China?",
    "How to make a cake?"
]
results = geogpt_batch_search(queries, topk=4, verbose=True)
```

### 配置模糊搜索
```python
# 方法1: 直接修改配置文件 fuzzy_search_config.py
# 方法2: 运行时动态配置
from fuzzy_search_config import update_fuzzy_config, print_fuzzy_config

# 查看当前配置
print_fuzzy_config()

# 更新配置
update_fuzzy_config(
    threshold=0.8,        # 提高相似度要求
    max_candidates=100,   # 增加检查的候选数量
    enabled=True         # 确保启用模糊搜索
)
```

## 配置参数说明

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FUZZY_SEARCH_ENABLED` | `True` | 是否启用模糊搜索 |
| `FUZZY_SIMILARITY_THRESHOLD` | `0.7` | 相似度阈值(0-1) |
| `FUZZY_MAX_CANDIDATES` | `50` | 最大检查候选数量 |

### 相似度阈值建议

- **0.5**: 宽松匹配，可能匹配到不太相关的查询
- **0.7**: 推荐值，平衡准确性和覆盖率
- **0.8**: 严格匹配，只匹配非常相似的查询
- **0.9**: 非常严格，几乎等同于精确匹配

## 状态码说明

搜索函数返回的状态码含义：

- `'exact_hit'`: 精确缓存命中
- `'fuzzy_hit'`: 模糊缓存命中
- `'new'`: 新查询，已从服务器获取并缓存
- `'not_found'`: 在知识库中未找到答案
- `'timeout'`: 服务器请求超时
- `'error'`: 发生错误

## 工作原理

### 1. 查询处理流程
```
用户查询 → 精确匹配检查 → 模糊匹配检查 → 服务器请求 → 缓存存储
```

### 2. 缓存结构
- **主缓存**: `geogpt_cache:{hash}` - 存储实际的问答内容
- **映射缓存**: `geogpt_query_map:{hash}` - 存储查询到缓存key的映射

### 3. 模糊匹配算法
1. 获取所有查询映射
2. 标准化查询字符串（转小写、去除多余空格）
3. 计算与已存储查询的相似度
4. 返回相似度最高且达到阈值的匹配结果

## 性能考虑

### 优化建议
1. **合理设置最大候选数量**: 避免检查过多缓存条目
2. **调整相似度阈值**: 根据实际需求平衡准确性和性能
3. **定期清理过期缓存**: 保持Redis数据库的整洁

### 性能指标
- 精确匹配: O(1) 时间复杂度
- 模糊匹配: O(n) 时间复杂度，n为检查的候选数量

## 示例场景

### 场景1: 中英文混合查询
```python
# 原始查询被缓存
result1, status1 = geogpt_search("中国的首都是什么？")
# status1 = 'new' (首次查询)

# 相似查询会匹配到缓存
result2, status2 = geogpt_search("中国首都是什么")  
# status2 = 'fuzzy_hit' (模糊匹配)
```

### 场景2: 英文查询变体
```python
# 原始查询
result1, status1 = geogpt_search("What is the capital of China?")

# 变体查询
result2, status2 = geogpt_search("What's the capital of China")
# 可能会匹配到原始查询的缓存结果
```

## 调试和监控

### 启用详细日志
```python
# 启用调试模式
result = geogpt_batch_search(
    queries, 
    DEBUG_GEOGPT_SEARCH=True, 
    verbose=True
)
```

### 监控统计信息
批量搜索会输出详细的统计信息：
- 总查询数量
- 精确命中率
- 模糊命中率
- 新请求率
- 错误率

## 故障排除

### 常见问题

1. **模糊匹配不工作**
   - 检查 `FUZZY_SEARCH_ENABLED` 是否为 `True`
   - 检查相似度阈值是否合适
   - 确认Redis连接正常

2. **相似度匹配不准确**
   - 调整 `FUZZY_SIMILARITY_THRESHOLD` 参数
   - 增加 `FUZZY_MAX_CANDIDATES` 检查更多候选

3. **性能问题**
   - 减少 `FUZZY_MAX_CANDIDATES` 值
   - 提高相似度阈值以减少匹配尝试

### 测试功能
```python
# 运行内置测试
python geogpt_topkcommon.py
```

这将执行完整的功能测试，包括相似度计算和实际搜索。

## 扩展功能

### 自定义相似度算法
您可以修改 `calculate_similarity` 函数来使用不同的相似度算法：

```python
def calculate_similarity(str1: str, str2: str) -> float:
    # 使用Jaccard相似度
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0
```

### 查询预处理
可以添加更复杂的查询标准化逻辑：

```python
def advanced_normalize_query(query: str) -> str:
    # 移除标点符号
    import string
    query = query.translate(str.maketrans('', '', string.punctuation))
    # 转换同义词
    synonyms = {'什么': '啥', 'how': 'how to'}
    for old, new in synonyms.items():
        query = query.replace(old, new)
    return query.lower().strip()
```

## 总结

这个模糊搜索功能提供了一个智能的缓存查询系统，能够：
- 保持原始数据的完整性
- 提供灵活的相似匹配
- 提高缓存命中率
- 减少不必要的服务器请求

通过合理配置参数，您可以在准确性和性能之间找到最佳平衡点。
