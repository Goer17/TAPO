import requests
from typing import Dict, Any, Tuple
import random
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed
import redis
import hashlib
import pickle
import uuid
import difflib
import re


def send_post_request(
    url: str, access_token: str, payload: Dict[str, Any], timeout: float = 60.0
) -> Dict[str, Any]:
    """
    执行带认证的JSON POST请求

    参数：
    url (str): 请求地址
    access_token (str): Bearer令牌
    payload (dict): 请求体数据
    timeout (float): 超时时间（秒）

    返回：
    dict: 包含状态码、响应头和响应数据的字典

    异常：
    requests.exceptions.RequestException: 封装所有请求异常
    """
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    try:
        # 发送请求并自动序列化JSON
        response = requests.post(
            url, headers=headers, json=payload, timeout=timeout  # 自动处理JSON序列化
        )

        # 检查HTTP状态码
        response.raise_for_status()

        # 智能解析响应内容
        try:
            response_data = response.json()
        except ValueError:
            response_data = response.text

        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "data": response_data,
        }

    except requests.exceptions.RequestException as e:
        error_info = {
            "error_type": type(e).__name__,
            "status_code": getattr(e.response, "status_code", None),
            "error_message": str(e),
            "request_payload": payload,  # 保留请求数据用于调试
        }
        raise requests.exceptions.RequestException(error_info) from e


# Redis配置，使用你的服务地址和端口
REDIS_HOST = '10.200.99.220'
REDIS_PORT = 32132
REDIS_DB = 0
REDIS_EXPIRE = 7 * 24 * 3600  # 7天

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


def make_cache_key(query, topk):
    key_str = f"{query}||{topk}"
    return "geogpt_cache:" + hashlib.md5(key_str.encode('utf-8')).hexdigest()


def geogpt_search(query, topk=2, DEBUG_GEOGPT_SEARCH=False, verbose=False):
    # 选择使用优化版本或标准版本的模糊搜索缓存
    if USE_OPTIMIZED_FUZZY_SEARCH:
        cached_result, cache_status = fuzzy_search_cache_optimized(query, topk, redis_client, verbose)
    else:
        cached_result, cache_status = fuzzy_search_cache(query, topk, redis_client, verbose)
    if cached_result is not None:
        if DEBUG_GEOGPT_SEARCH and verbose:
            print(f"[DEBUG] {cache_status} for: {query} topk={topk}")
        return cached_result, cache_status
    try:
        request_payload = {
            "session_id": str(uuid.uuid4()),  # 每次生成随机uuid
            "query": query,
            "rag_switch": "chat_common",
            "top_k": topk,
        }
        # TODO(wenxun): Only for test, remove it in production
        result = send_post_request(
            url="http://10.200.48.226:30773/plugin/rag/retrieve/geo_common",
            access_token="sk-K30WQxD91JJ4mQLHMM5Q",
            payload=request_payload,
        )
        if DEBUG_GEOGPT_SEARCH:
            if random.random() < 0.01:
                print(f"[DEBUG] The query is: {query}")
                pprint(result)
        cacheit = True
        data = result.get("data", {})
        if not isinstance(data, dict):
            data = {}
            cacheit = False
        final_list = data.get("final", [])
        all_page_contents = []
        for item in final_list:
            page_content = item.get("page_content", None)
            temp_metadata = item.get("metadata", {})
            title = temp_metadata.get("title", None)
            if page_content:
                all_page_contents.append(f"(Title: {title}) {page_content}\n")
        if DEBUG_GEOGPT_SEARCH and len(final_list) == 0 and verbose:
            print(f"[ERROR] Items without page_content for query '{query}':")
            print("Request payload:")
            pprint(request_payload)
            print("Response data:")
            pprint(result)
        format_reference = ""
        for i, content in enumerate(all_page_contents):
            format_reference += f"Doc {i + 1}{content}"
        # 写入缓存
        if cacheit:
            try:
                cache_key = make_cache_key(query, topk)
                redis_client.set(cache_key, pickle.dumps(format_reference), ex=REDIS_EXPIRE)
                # 同时存储查询映射，用于模糊搜索
                store_query_mapping(query, topk, cache_key, redis_client)
            except Exception as e:
                print(f"[WARN] Redis set error: {e}")

        if format_reference:
            return format_reference, 'new'
        else:
            return f"Sorry, not found answer for question {query} in the knowledge base.", 'not_found'
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        return "Sorry, Server unavailable. Please try again later.", 'timeout'
    except Exception as other_err:
        print(f"请求失败: {other_err}")
        return "Sorry, something went wrong. Please try again later.", 'error'


def geogpt_batch_search(queries, topk=2, DEBUG_GEOGPT_SEARCH=True, max_workers=8, verbose=False):
    """
    并发批量搜索函数
    """
    nonsense_count = 0
    exact_hit_count = 0
    fuzzy_hit_count = 0
    timeout_count = 0
    new_count = 0
    not_found_count = 0
    error_count = 0
    results = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, query in enumerate(queries):
            if not query or query.strip() in ["", "...", "and", "query"]:
                if verbose:
                    print("[WARN] Asking for searching nonsense...")
                results[idx] = (
                    "Sorry, I don't think the question you asked between <search> and </search> is meaningful. Please try again."
                )
                nonsense_count += 1
            elif len(query) < 3:
                if verbose:
                    print("[WARN] Asking for searching nonsense, too short query...")
                results[idx] = (
                    "Sorry, I don't think the question you asked between <search> and </search> is meaningful (it is too short). Please try again."
                )
                nonsense_count += 1
            elif len(query) > 300:
                if verbose:
                    print("[WARN] Asking for searching nonsense, too long query...")
                results[idx] = (
                    "Sorry, I don't think the question you asked between <search> and </search> is meaningful (it is too long). Please try again."
                )
                nonsense_count += 1
            else:
                future = executor.submit(
                    geogpt_search, query, topk, False, verbose
                )
                future_to_idx[future] = idx
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result, status = future.result()
                results[idx] = result
                if status == 'exact_hit':
                    exact_hit_count += 1
                elif status == 'fuzzy_hit':
                    fuzzy_hit_count += 1
                elif status == 'new':
                    new_count += 1
                elif status == 'not_found':
                    not_found_count += 1
                elif status == 'timeout':
                    timeout_count += 1
                elif status == 'error':
                    error_count += 1
            except Exception as e:
                results[idx] = f"Error: {e}"
                error_count += 1
    if DEBUG_GEOGPT_SEARCH:
        len_queries = len(queries)
        len_results = len(results)
        total_cache_hits = exact_hit_count + fuzzy_hit_count
        print("="*50)
        print(f"Total queries: {len_queries}")
        print(f"Total results: {len_results}")
        print(f"Nonsense: {nonsense_count / len_queries * 100:.2f}%")
        print(f"Exact cache hits: {exact_hit_count / len_queries * 100:.2f}%")
        print(f"Fuzzy cache hits: {fuzzy_hit_count / len_queries * 100:.2f}%")
        print(f"Total cache hits: {total_cache_hits / len_queries * 100:.2f}%")
        print(f"Timeout/Error: {(timeout_count + error_count) / len_queries * 100:.2f}%")
        if new_count > 0:
            print(f"T/E Rate (Server side): {(timeout_count + error_count) / new_count * 100:.2f}%")
        print(f"Newly requested: {new_count / len_queries * 100:.2f}%")
        print(f"Not found: {not_found_count / len_queries * 100:.2f}%")
        if len_queries != len_results:
            print(f"[ERROR] Mismatch in queries and results length: {len_queries} vs {len_results}")
        print("="*50)
    return results


# 模糊搜索配置
FUZZY_SEARCH_ENABLED = True
FUZZY_SIMILARITY_THRESHOLD = 0.9  # 相似度阈值(0-1)
FUZZY_MAX_CANDIDATES = 200  # 增加候选数量，提高匹配质量
FUZZY_SCAN_COUNT = 100  # 每次SCAN操作的数量
USE_OPTIMIZED_FUZZY_SEARCH = True  # 是否使用优化版本的模糊搜索


def calculate_similarity(str1: str, str2: str) -> float:
    """
    计算两个字符串的相似度
    使用Python内置的difflib库
    返回值范围 0-1，1表示完全相同
    """
    return difflib.SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def normalize_query(query: str) -> str:
    """
    标准化查询字符串，用于模糊匹配
    """
    # 转换为小写
    query = query.lower()
    # 移除多余的空格
    query = re.sub(r'\s+', ' ', query).strip()
    # 移除标点符号（可选，根据需要调整）
    # query = re.sub(r'[^\w\s]', '', query)
    return query


def fuzzy_search_cache(query: str, topk: int, redis_client, verbose: bool = False) -> Tuple[str, str]:
    """
    在Redis缓存中执行模糊搜索
    返回: (结果内容, 状态)
    状态可能是: 'exact_hit', 'fuzzy_hit', 'not_found'
    """
    # 首先尝试精确匹配
    exact_cache_key = make_cache_key(query, topk)
    try:
        cached = redis_client.get(exact_cache_key)
        if cached is not None:
            if verbose:
                print(f"[DEBUG] Exact cache hit for: {query}")
            return pickle.loads(cached), 'exact_hit'
    except Exception as e:
        if verbose:
            print(f"[WARN] Redis exact get error: {e}")
    
    # 如果没有精确匹配且启用了模糊搜索，则进行模糊搜索
    if not FUZZY_SEARCH_ENABLED:
        return None, 'not_found'
    
    try:
        normalized_query = normalize_query(query)
        best_match_key = None
        best_similarity = 0
        
        # 使用SCAN代替KEYS，避免阻塞Redis
        # 获取所有查询映射的keys用于模糊搜索
        query_keys = []
        cursor = 0
        query_to_key_pattern = "geogpt_query_map:*"
        
        while True:
            cursor, keys = redis_client.scan(cursor=cursor, match=query_to_key_pattern, count=100)
            query_keys.extend(keys)
            
            # 限制检查的key数量，避免性能问题
            if len(query_keys) >= FUZZY_MAX_CANDIDATES:
                query_keys = query_keys[:FUZZY_MAX_CANDIDATES]
                break
                
            if cursor == 0:
                break
        
        if not query_keys:
            return None, 'not_found'
        
        # 如果候选项很多，使用随机采样提高覆盖率
        if len(query_keys) > FUZZY_MAX_CANDIDATES:
            query_keys = random.sample(query_keys, FUZZY_MAX_CANDIDATES)
        
        for query_key in query_keys:
            try:
                stored_info = redis_client.get(query_key)
                if stored_info:
                    query_info = pickle.loads(stored_info)
                    stored_query = query_info.get('query', '')
                    stored_topk = query_info.get('topk', 0)
                    
                    # 只比较相同topk的查询
                    if stored_topk == topk:
                        normalized_stored = normalize_query(stored_query)
                        similarity = calculate_similarity(normalized_query, normalized_stored)
                        
                        if similarity > best_similarity and similarity >= FUZZY_SIMILARITY_THRESHOLD:
                            best_similarity = similarity
                            best_match_key = query_info.get('cache_key')
            except Exception as e:
                if verbose:
                    print(f"[WARN] Error processing query key {query_key}: {e}")
                continue
        
        # 如果找到了足够相似的查询，返回其缓存结果
        if best_match_key:
            try:
                cached_result = redis_client.get(best_match_key)
                if cached_result:
                    if verbose:
                        print(f"[DEBUG] Fuzzy cache hit for: {query} (similarity: {best_similarity:.3f})")
                    return pickle.loads(cached_result), 'fuzzy_hit'
            except Exception as e:
                if verbose:
                    print(f"[WARN] Error retrieving fuzzy match: {e}")
        
    except Exception as e:
        if verbose:
            print(f"[WARN] Fuzzy search error: {e}")
    
    return None, 'not_found'


def fuzzy_search_cache_optimized(query: str, topk: int, redis_client, verbose: bool = False) -> Tuple[str, str]:
    """
    优化版本的模糊搜索，使用Redis Sorted Set提高性能
    当数据量很大时，这个版本会更高效
    """
    # 首先尝试精确匹配
    exact_cache_key = make_cache_key(query, topk)
    try:
        cached = redis_client.get(exact_cache_key)
        if cached is not None:
            if verbose:
                print(f"[DEBUG] Exact cache hit for: {query}")
            return pickle.loads(cached), 'exact_hit'
    except Exception as e:
        if verbose:
            print(f"[WARN] Redis exact get error: {e}")
    
    if not FUZZY_SEARCH_ENABLED:
        return None, 'not_found'
    
    try:
        normalized_query = normalize_query(query)
        
        # 使用Redis Sorted Set存储查询的n-gram特征
        # 这样可以更快地找到相似的查询
        query_ngrams = set(get_ngrams(normalized_query, 3))  # 3-gram
        
        best_match_key = None
        best_similarity = 0
        
        # 使用pipeline批量获取，提高性能
        pipe = redis_client.pipeline()
        
        # 分批处理查询映射
        cursor = 0
        batch_size = 50
        processed_count = 0
        
        while processed_count < FUZZY_MAX_CANDIDATES:
            cursor, keys = redis_client.scan(
                cursor=cursor, 
                match="geogpt_query_map:*", 
                count=FUZZY_SCAN_COUNT
            )
            
            if not keys:
                if cursor == 0:
                    break
                continue
            
            # 批量获取查询信息
            for key in keys[:min(batch_size, FUZZY_MAX_CANDIDATES - processed_count)]:
                pipe.get(key)
            
            results = pipe.execute()
            pipe.reset()
            
            # 处理这一批的结果
            for i, stored_info in enumerate(results):
                if stored_info and processed_count < FUZZY_MAX_CANDIDATES:
                    try:
                        query_info = pickle.loads(stored_info)
                        stored_query = query_info.get('query', '')
                        stored_topk = query_info.get('topk', 0)
                        
                        if stored_topk == topk:
                            normalized_stored = normalize_query(stored_query)
                            
                            # 快速预筛选：使用n-gram重叠度
                            stored_ngrams = set(get_ngrams(normalized_stored, 3))
                            if query_ngrams and stored_ngrams:
                                ngram_overlap = len(query_ngrams & stored_ngrams) / len(query_ngrams | stored_ngrams)
                                # 只有n-gram重叠度足够高才进行详细相似度计算
                                if ngram_overlap >= 0.3:
                                    similarity = calculate_similarity(normalized_query, normalized_stored)
                                    if similarity > best_similarity and similarity >= FUZZY_SIMILARITY_THRESHOLD:
                                        best_similarity = similarity
                                        best_match_key = query_info.get('cache_key')
                    except Exception as e:
                        if verbose:
                            print(f"[WARN] Error processing query info: {e}")
                    
                    processed_count += 1
            
            if cursor == 0:
                break
        
        # 如果找到了足够相似的查询，返回其缓存结果
        if best_match_key:
            try:
                cached_result = redis_client.get(best_match_key)
                if cached_result:
                    if verbose:
                        print(f"[DEBUG] Optimized fuzzy cache hit for: {query} (similarity: {best_similarity:.3f})")
                    return pickle.loads(cached_result), 'fuzzy_hit'
            except Exception as e:
                if verbose:
                    print(f"[WARN] Error retrieving fuzzy match: {e}")
        
    except Exception as e:
        if verbose:
            print(f"[WARN] Optimized fuzzy search error: {e}")
    
    return None, 'not_found'


def get_ngrams(text: str, n: int = 3):
    """
    生成文本的n-gram特征
    用于快速相似度预筛选
    """
    if len(text) < n:
        return [text]
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def store_query_mapping(query: str, topk: int, cache_key: str, redis_client):
    """
    存储查询到缓存key的映射，用于后续的模糊搜索
    """
    try:
        mapping_key = "geogpt_query_map:" + hashlib.md5(f"{query}||{topk}".encode('utf-8')).hexdigest()
        query_info = {
            'query': query,
            'topk': topk,
            'cache_key': cache_key
        }
        redis_client.set(mapping_key, pickle.dumps(query_info), ex=REDIS_EXPIRE)
    except Exception as e:
        print(f"[WARN] Error storing query mapping: {e}")


def test_fuzzy_search():
    """
    测试模糊搜索功能的示例函数
    """
    print("=== 模糊搜索功能测试 ===")
    
    # 测试相似度计算
    test_queries = [
        ("中国的首都是什么？", "中国首都是什么"),
        ("What is the capital of China?", "What's the capital of China"),
        ("How to make a cake?", "如何制作蛋糕？"),
        ("Google presentation end screen", "Google presentation end screen mark")
    ]
    
    print("\n相似度计算测试:")
    for q1, q2 in test_queries:
        similarity = calculate_similarity(q1, q2)
        print(f"'{q1}' vs '{q2}': {similarity:.3f}")
    
    print("\n当前模糊搜索配置:")
    print(f"- 启用状态: {FUZZY_SEARCH_ENABLED}")
    print(f"- 相似度阈值: {FUZZY_SIMILARITY_THRESHOLD}")
    print(f"- 最大候选数量: {FUZZY_MAX_CANDIDATES}")
    print(f"- SCAN批次大小: {FUZZY_SCAN_COUNT}")
    print(f"- 使用优化版本: {USE_OPTIMIZED_FUZZY_SEARCH}")
    
    # 测试n-gram特征提取
    print("\nn-gram特征测试:")
    test_text = "How to make a cake"
    ngrams = get_ngrams(test_text, 3)
    print(f"文本: '{test_text}'")
    print(f"3-grams: {ngrams[:10]}...")  # 只显示前10个


# 使用示例
if __name__ == "__main__":
    # 首先测试模糊搜索功能
    test_fuzzy_search()
    
    print("\n" + "="*50)
    print("开始实际搜索测试...")
    
    queries = [
        "中国的首都是什么？",
        "What is the capital of China?",
        "How to make a cake?",
        "Google presentation end screen mark",
        "When did the first £1 coin come out?",
        "Python编程语言的特点是什么？",
        "What are the benefits of machine learning?",
        "如何提高工作效率？",
        "Best practices for database design",
        "JavaScript异步编程怎么做？",
        "How to configure Docker containers?",
        "什么是人工智能的发展趋势？",
        "React vs Vue.js comparison",
        "如何学习数据结构和算法？",
        "Cloud computing security concerns",
        "MySQL数据库优化技巧",
        "How to implement REST API?",
        "移动端开发框架选择",
        "Git version control best practices",
        "深度学习模型训练技巧",
        "How to deploy applications to AWS?",
        "区块链技术的应用场景",
        "Microservices architecture patterns",
        "如何进行代码重构？",
        "Kubernetes cluster management"
    ]
    results = geogpt_batch_search(queries, topk=3)
    for i, query in enumerate(queries):
        print(f"Query: {query}")
        print(f"Result: {results[i][:100]}")
        print("=" * 50)
        
    # 测试模糊匹配 - 运行相似的查询
    print("\n" + "="*50)
    print("模糊匹配测试 - 使用相似但不完全相同的查询...")
    
    similar_queries = [
        "中国首都是什么",  # 与第一个查询相似
        "What's the capital of China",  # 与第二个查询相似  
        "How can I make a cake",  # 与第三个查询相似
        "Python编程有什么特点",  # 与Python特点查询相似
        "What are machine learning advantages?",  # 与机器学习好处查询相似
        "怎样提高工作效率",  # 与提高工作效率查询相似
        "Database design best practices",  # 与数据库设计查询相似
        "JavaScript异步编程方法",  # 与JS异步编程查询相似
        "How to setup Docker containers?",  # 与Docker配置查询相似
        "AI发展趋势是什么",  # 与AI发展趋势查询相似
        "React和Vue.js对比",  # 与React vs Vue查询相似
        "怎么学数据结构算法",  # 与学习数据结构算法查询相似
        "Cloud security concerns",  # 与云计算安全查询相似
        "MySQL优化方法",  # 与MySQL优化查询相似
        "How to build REST API?",  # 与REST API实现查询相似
        "移动开发框架怎么选",  # 与移动端框架选择查询相似
        "Git best practices",  # 与Git最佳实践查询相似
        "深度学习训练方法",  # 与深度学习训练查询相似
        "How to deploy to AWS?",  # 与AWS部署查询相似
        "区块链应用有哪些",  # 与区块链应用查询相似
        "Microservices patterns",  # 与微服务架构查询相似
        "代码重构怎么做",  # 与代码重构查询相似
        "Kubernetes management"  # 与Kubernetes管理查询相似
    ]
    
    similar_results = geogpt_batch_search(similar_queries, topk=3, verbose=True)
    for i, query in enumerate(similar_queries):
        print(f"Similar Query: {query}")
        print(f"Result: {similar_results[i][:100]}")
        print("=" * 50)
    
    # 测试边界情况和特殊查询
    print("\n" + "="*50)
    print("边界情况测试...")
    
    edge_case_queries = [
        "",  # 空查询
        "   ",  # 只有空格
        "a",  # 单字符
        "What?",  # 非常短的查询
        "这是一个非常非常非常长的查询字符串，用来测试系统在处理超长查询时的性能表现和稳定性如何",  # 超长查询
        "🤔🔍💡",  # emoji查询
        "SELECT * FROM users WHERE id = 1;",  # SQL查询
        "https://www.example.com/api/v1/users",  # URL查询
        "2024-12-31 23:59:59",  # 时间格式
        "user@example.com",  # 邮箱格式
    ]
    
    edge_results = geogpt_batch_search(edge_case_queries, topk=3, verbose=True)
    for i, query in enumerate(edge_case_queries):
        print(f"Edge Case Query: '{query}'")
        print(f"Result: {edge_results[i][:100]}...")  # 只显示前100个字符
        print("-" * 30)
    
    # 性能压力测试
    print("\n" + "="*50)
    print("性能压力测试 - 大批量查询...")
    
    import time
    
    # 生成大量重复和相似的查询来测试缓存效果
    stress_queries = []
    base_queries = [
        "What is artificial intelligence?",
        "How to learn programming?",  
        "什么是机器学习？",
        "数据库设计原则",
        "Cloud computing benefits"
    ]
    
    # 为每个基础查询生成多个变体
    variations = [
        "",
        " ",
        "?",
        "？",
        " please",
        " exactly",
        "can you tell me ",
        "I want to know ",
        "请告诉我",
        "我想了解"
    ]
    
    for base_query in base_queries:
        stress_queries.append(base_query)  # 原查询
        for variation in variations:
            if base_query.startswith("什么") or base_query.startswith("数据"):
                # 中文查询加中文变体
                if variation in ["请告诉我", "我想了解"]:
                    stress_queries.append(variation + base_query)
            else:
                # 英文查询加英文变体
                if variation in ["can you tell me ", "I want to know "]:
                    stress_queries.append(variation + base_query)
                else:
                    stress_queries.append(base_query + variation)
    
    print(f"生成了 {len(stress_queries)} 个测试查询")
    
    # 执行压力测试
    start_time = time.time()
    stress_results = geogpt_batch_search(stress_queries, topk=3, max_workers=16, verbose=False)
    end_time = time.time()
    
    print(f"压力测试完成，耗时: {end_time - start_time:.2f}秒")
    print(f"平均每个查询耗时: {(end_time - start_time) / len(stress_queries):.3f}秒")
    print(f"QPS (每秒查询数): {len(stress_queries) / (end_time - start_time):.2f}")
