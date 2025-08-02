import requests
from typing import Dict, Any, Tuple
import random
from pprint import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed
import redis
import hashlib
import pickle
import difflib
import re
import time
import os


def send_post_request(
    url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float = 60.0
) -> Dict[str, Any]:
    """
    执行带自定义头部的JSON POST请求

    参数：
    url (str): 请求地址
    headers (dict): 请求头
    payload (dict): 请求体数据
    timeout (float): 超时时间（秒）

    返回：
    dict: 包含状态码、响应头和响应数据的字典

    异常：
    requests.exceptions.RequestException: 封装所有请求异常
    """
    try:
        # 发送请求并自动序列化JSON
        response = requests.post(
            url, headers=headers, json=payload, timeout=timeout
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
REDIS_HOST = os.environ.get('SEARCH_REDIS_HOST', '10.200.99.220')
REDIS_PORT = int(os.environ.get('SEARCH_REDIS_PORT', 32132))
REDIS_DB = 0
REDIS_EXPIRE = 7 * 24 * 3600  # 7天

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, socket_connect_timeout=10)

try:
    redis_client.ping()  # 主动测试连接
except redis.exceptions.ConnectionError as e:
    print(f"Redis连接失败: {e}")

# Serper API
DEFAULT_SERPER_API_KEY = 'your_serper_api_key'
SERPER_API_URL = "https://google.serper.dev/search"

# 模糊搜索配置
FUZZY_SEARCH_ENABLED = True
FUZZY_SIMILARITY_THRESHOLD = 0.9  # 相似度阈值(0-1)
FUZZY_MAX_CANDIDATES = 200  # 增加候选数量，提高匹配质量
FUZZY_SCAN_COUNT = 100  # 每次SCAN操作的数量
USE_OPTIMIZED_FUZZY_SEARCH = True  # 是否使用优化版本的模糊搜索


def make_cache_key(query, topk):
    """生成缓存key"""
    key_str = f"{query}||{topk}"
    return "serper_cache:" + hashlib.md5(key_str.encode('utf-8')).hexdigest()


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
        mapping_key = "serper_query_map:" + hashlib.md5(f"{query}||{topk}".encode('utf-8')).hexdigest()
        query_info = {
            'query': query,
            'topk': topk,
            'cache_key': cache_key
        }
        redis_client.set(mapping_key, pickle.dumps(query_info), ex=REDIS_EXPIRE)
    except Exception as e:
        print(f"[WARN] Error storing query mapping: {e}")


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
        query_to_key_pattern = "serper_query_map:*"
        
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
                match="serper_query_map:*", 
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


def serper_search(query: str, topk: int = 5, api_key: str = DEFAULT_SERPER_API_KEY, 
                 DEBUG_SERPER_SEARCH: bool = False, verbose: bool = False, timeout: float = 30.0) -> Tuple[str, str]:
    """
    使用Google Serper API进行搜索，带缓存和模糊搜索功能
    
    参数：
    query (str): 搜索查询
    topk (int): 返回的最大结果数量
    api_key (str): Serper API密钥
    DEBUG_SERPER_SEARCH (bool): 是否启用调试模式
    verbose (bool): 是否显示详细信息
    timeout (float): 超时时间（秒）
    
    返回：
    tuple: (格式化的搜索结果, 状态)
    状态可能是: 'exact_hit', 'fuzzy_hit', 'new', 'not_found', 'timeout', 'error'
    """
    # 选择使用优化版本或标准版本的模糊搜索缓存
    if USE_OPTIMIZED_FUZZY_SEARCH:
        cached_result, cache_status = fuzzy_search_cache_optimized(query, topk, redis_client, verbose)
    else:
        cached_result, cache_status = fuzzy_search_cache(query, topk, redis_client, verbose)
    
    if cached_result is not None:
        if DEBUG_SERPER_SEARCH and verbose:
            print(f"[DEBUG] {cache_status} for: {query} topk={topk}")
        return cached_result, cache_status
    
    try:
        payload = {
            "q": query,
            "num": topk  # Serper API使用num参数控制结果数量
        }
        
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }
        
        result = send_post_request(
            url=SERPER_API_URL,
            headers=headers,
            payload=payload,
            timeout=timeout
        )
        
        if DEBUG_SERPER_SEARCH:
            if random.random() < 0.01:
                print(f"[DEBUG] The query is: {query}")
                pprint(result)
        
        cacheit = True
        data = result.get("data", {})
        if not isinstance(data, dict):
            data = {}
            cacheit = False
        
        # 格式化Serper搜索结果
        format_reference = format_serper_results_for_cache(data, topk)
        
        if DEBUG_SERPER_SEARCH and not format_reference and verbose:
            print(f"[ERROR] No results for query '{query}':")
            print("Request payload:")
            pprint(payload)
            print("Response data:")
            pprint(result)
        
        # 写入缓存
        if cacheit and format_reference:
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
            return f"Sorry, not found answer for question '{query}' in the search results.", 'not_found'
            
    except requests.exceptions.RequestException as e:
        print(f"搜索请求失败: {e}")
        return "Sorry, Search service unavailable. Please try again later.", 'timeout'
    except Exception as other_err:
        print(f"搜索失败: {other_err}")
        return "Sorry, something went wrong with the search. Please try again later.", 'error'


def format_serper_results_for_cache(data: Dict[str, Any], max_results: int = 5) -> str:
    """
    格式化Serper搜索结果为缓存用的文本格式
    
    参数：
    data (dict): Serper API返回的数据部分
    max_results (int): 最多返回的结果数量
    
    返回：
    str: 格式化后的搜索结果文本
    """
    if not data:
        return ""
    
    formatted_results = []
    
    # 添加知识面板信息（如果有）
    knowledge_graph = data.get("knowledgeGraph", {})
    if knowledge_graph:
        title = knowledge_graph.get("title", "")
        description = knowledge_graph.get("description", "")
        if title and description:
            kg_info = f"(Knowledge Panel: {title}) {description}\n"
            formatted_results.append(kg_info)
    
    # 处理有机搜索结果
    organic_results = data.get("organic", [])
    for i, result in enumerate(organic_results[:max_results]):
        title = result.get("title", "No Title")
        snippet = result.get("snippet", "No Description")

        formatted_result = f"(Title: {title}) {snippet}\n"
        formatted_results.append(formatted_result)
    
    # 组合所有结果
    final_reference = ""
    for i, content in enumerate(formatted_results):
        final_reference += f"Doc {i + 1}: {content}"
    
    return final_reference


def format_serper_results(search_result: Dict[str, Any], max_results: int = 5) -> str:
    """
    格式化Serper搜索结果为可读的文本格式（用于显示）
    
    参数：
    search_result (dict): Serper API返回的完整搜索结果
    max_results (int): 最多返回的结果数量
    
    返回：
    str: 格式化后的搜索结果文本
    """
    if "error" in search_result:
        return f"搜索出错: {search_result['error']}"
    
    data = search_result.get("data", {})
    if not data:
        return "没有找到搜索结果"
    
    formatted_results = []
    
    # 处理有机搜索结果
    organic_results = data.get("organic", [])
    for i, result in enumerate(organic_results[:max_results]):
        title = result.get("title", "无标题")
        snippet = result.get("snippet", "无描述")
        
        formatted_result = f"结果 {i + 1}:\n"
        formatted_result += f"标题: {title}\n"
        formatted_result += f"描述: {snippet}\n"
        formatted_result += f"链接: {result.get('link', '')}\n"
        formatted_results.append(formatted_result)
    
    # 添加知识面板信息（如果有）
    knowledge_graph = data.get("knowledgeGraph", {})
    if knowledge_graph:
        title = knowledge_graph.get("title", "")
        description = knowledge_graph.get("description", "")
        if title and description:
            kg_info = f"\n知识面板:\n标题: {title}\n描述: {description}\n"
            formatted_results.insert(0, kg_info)
    
    # 添加相关问题（如果有）
    people_also_ask = data.get("peopleAlsoAsk", [])
    if people_also_ask:
        related_questions = "\n相关问题:\n"
        for i, question in enumerate(people_also_ask[:3]):  # 只显示前3个
            related_questions += f"{i + 1}. {question.get('question', '')}\n"
        formatted_results.append(related_questions)
    
    return "\n".join(formatted_results) if formatted_results else "没有找到相关结果"


def serper_batch_search(queries: list, topk: int = 5, api_key: str = DEFAULT_SERPER_API_KEY,
                       DEBUG_SERPER_SEARCH: bool = True, max_workers: int = 8, verbose: bool = False,
                       timeout: float = 30.0) -> list:
    """
    并发批量Serper搜索函数
    
    参数：
    queries (list): 搜索查询列表
    topk (int): 每个查询返回的最大结果数量
    api_key (str): Serper API密钥
    DEBUG_SERPER_SEARCH (bool): 是否启用调试模式
    max_workers (int): 最大并发数
    verbose (bool): 是否显示详细信息
    timeout (float): 超时时间（秒）
    
    返回：
    list: 搜索结果列表
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
                    "Sorry, I don't think the question you asked is meaningful. Please try again."
                )
                nonsense_count += 1
            elif len(query) < 3:
                if verbose:
                    print("[WARN] Asking for searching nonsense, too short query...")
                results[idx] = (
                    "Sorry, I don't think the question you asked is meaningful (it is too short). Please try again."
                )
                nonsense_count += 1
            elif len(query) > 300:
                if verbose:
                    print("[WARN] Asking for searching nonsense, too long query...")
                results[idx] = (
                    "Sorry, I don't think the question you asked is meaningful (it is too long). Please try again."
                )
                nonsense_count += 1
            else:
                future = executor.submit(
                    serper_search, query, topk, api_key, False, verbose, timeout
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
    
    if new_count > 0:
        try:
            redis_client.incrby('tcg:external_search_count', new_count)
        except Exception as e:
            print(f"[WARN] Error updating external search count in Redis: {e}")

    if DEBUG_SERPER_SEARCH:
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


def test_fuzzy_search():
    """
    测试模糊搜索功能的示例函数
    """
    print("=== Serper模糊搜索功能测试 ===")
    
    # 测试相似度计算
    test_queries = [
        ("中国的首都是什么？", "中国首都是什么"),
        ("What is the capital of China?", "What's the capital of China"),
        ("How to make a cake?", "如何制作蛋糕？"),
        ("Python programming tutorial", "Python programming guide")
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


def test_serper_search():
    """
    测试Serper搜索功能
    """
    print("=== Serper搜索功能测试 ===")
    
    # 单个搜索测试
    test_query = "apple inc"
    print(f"测试查询: {test_query}")
    
    try:
        result, status = serper_search(test_query, topk=3, verbose=True)
        print(f"搜索成功! 状态: {status}")
        print("搜索结果:")
        print(result[:500] + "..." if len(result) > 500 else result)
        
    except Exception as e:
        print(f"搜索失败: {e}")
    
    print("\n" + "="*50)
    
    # 批量搜索测试
    test_queries = [
        "python programming",
        "machine learning",
        "artificial intelligence",
        "web development",
        "data science"
    ]
    
    print("批量搜索测试:")
    try:
        batch_results = serper_batch_search(test_queries, topk=3, max_workers=3, verbose=True)
        
        print("\n批量搜索结果概览:")
        for i, (query, result) in enumerate(zip(test_queries, batch_results)):
            if isinstance(result, str) and not result.startswith("Error:"):
                print(f"{i+1}. {query}: 搜索成功")
            else:
                print(f"{i+1}. {query}: 搜索失败 - {result}")
                
    except Exception as e:
        print(f"批量搜索失败: {e}")


# 使用示例
if __name__ == "__main__":
    # 首先测试模糊搜索功能
    test_fuzzy_search()
    
    print("\n" + "="*50)
    print("开始Serper搜索测试...")
    
    # 基础搜索测试
    queries = [
        "What is artificial intelligence?",
        "How to learn Python programming?",
        "机器学习算法有哪些？",
        "Best web development frameworks 2025",
        "What are the benefits of cloud computing?",
        "How to optimize database performance?",
        "React vs Vue.js comparison",
        "什么是区块链技术？",
        "Docker containerization tutorial",
        "How to deploy applications on AWS?"
    ]
    
    results = serper_batch_search(queries, topk=3, DEBUG_SERPER_SEARCH=True, verbose=False)
    for i, query in enumerate(queries):
        print(f"Query: {query}")
        # result_preview = results[i][:100] + "..." if len(results[i]) > 100 else results[i]
        result_preview = results[i]
        print(f"Result: {result_preview}")
        print("=" * 50)
        
    # 测试模糊匹配 - 运行相似的查询
    print("\n" + "="*50)
    print("模糊匹配测试 - 使用相似但不完全相同的查询...")
    
    similar_queries = [
        "What's artificial intelligence?",  # 与第一个查询相似
        "How can I learn Python programming",  # 与第二个查询相似  
        "机器学习有什么算法",  # 与第三个查询相似
        "Best frameworks for web development",  # 与第四个查询相似
        "What are cloud computing benefits?",  # 与第五个查询相似
        "How to improve database performance?",  # 与第六个查询相似
        "React和Vue.js对比",  # 与第七个查询相似
        "区块链技术是什么",  # 与第八个查询相似
        "Docker container tutorial",  # 与第九个查询相似
        "How to deploy apps to AWS?"  # 与第十个查询相似
    ]
    
    similar_results = serper_batch_search(similar_queries, topk=3, verbose=True, DEBUG_SERPER_SEARCH=True)
    for i, query in enumerate(similar_queries):
        print(f"Similar Query: {query}")
        result_preview = similar_results[i][:100] + "..." if len(similar_results[i]) > 100 else similar_results[i]
        print(f"Result: {result_preview}")
        print("=" * 50)
    
    # 测试边界情况和特殊查询
    print("\n" + "="*50)
    print("边界情况测试...")
    
    edge_case_queries = [
        "",  # 空查询
        "   ",  # 只有空格
        "a",  # 单字符
        "What?",  # 非常短的查询
        "这是一个非常非常非常长的查询字符串，用来测试系统在处理超长查询时的性能表现和稳定性，看看搜索引擎如何处理这种情况",  # 超长查询
        "🤔🔍💡",  # emoji查询
        "python OR java",  # 布尔查询
        "\"exact phrase search\"",  # 精确短语搜索
        "site:github.com python",  # 站点限制搜索
        "filetype:pdf machine learning"  # 文件类型搜索
    ]
    
    edge_results = serper_batch_search(edge_case_queries, topk=3, verbose=True, DEBUG_SERPER_SEARCH=True)
    for i, query in enumerate(edge_case_queries):
        print(f"Edge Case Query: '{query}'")
        result_preview = edge_results[i][:100] + "..." if len(edge_results[i]) > 100 else edge_results[i]
        print(f"Result: {result_preview}")
        print("-" * 30)
    
    # 性能压力测试
    print("\n" + "="*50)
    print("性能压力测试 - 大批量查询...")
    
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
        " 2025",
        " tutorial",
        "how to ",
        "what is ",
        "如何",
        "什么是"
    ]
    
    for base_query in base_queries:
        stress_queries.append(base_query)  # 原查询
        for variation in variations:
            if base_query.startswith("什么") or base_query.startswith("数据"):
                # 中文查询加中文变体
                if variation in ["如何", "什么是"]:
                    stress_queries.append(variation + base_query[2:])  # 去掉原有的"什么"
            else:
                # 英文查询加英文变体
                if variation in ["how to ", "what is "]:
                    stress_queries.append(variation + base_query.lower())
                else:
                    stress_queries.append(base_query + variation)
    
    print(f"生成了 {len(stress_queries)} 个测试查询")
    
    # 执行压力测试
    start_time = time.time()
    stress_results = serper_batch_search(stress_queries, topk=3, max_workers=16, verbose=False, DEBUG_SERPER_SEARCH=True)
    end_time = time.time()
    
    print(f"压力测试完成，耗时: {end_time - start_time:.2f}秒")
    print(f"平均每个查询耗时: {(end_time - start_time) / len(stress_queries):.3f}秒")
    print(f"QPS (每秒查询数): {len(stress_queries) / (end_time - start_time):.2f}")
