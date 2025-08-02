FUZZY_SEARCH_ENABLED = True

FUZZY_SIMILARITY_THRESHOLD = 0.7

FUZZY_MAX_CANDIDATES = 50

REDIS_HOST = '10.200.99.220'
REDIS_PORT = 32132
REDIS_DB = 0
REDIS_EXPIRE = 7 * 24 * 3600

DEBUG_GEOGPT_SEARCH = False
DEFAULT_VERBOSE = False

REMOVE_PUNCTUATION_IN_COMPARISON = False

ENABLE_QUERY_NORMALIZATION = True

DEFAULT_MAX_WORKERS = 8
DEFAULT_TOPK = 2

SIMILARITY_METHOD = 'difflib'

def get_fuzzy_config():
    return {
        'enabled': FUZZY_SEARCH_ENABLED,
        'threshold': FUZZY_SIMILARITY_THRESHOLD,
        'max_candidates': FUZZY_MAX_CANDIDATES,
        'similarity_method': SIMILARITY_METHOD,
        'normalize_query': ENABLE_QUERY_NORMALIZATION,
        'remove_punctuation': REMOVE_PUNCTUATION_IN_COMPARISON
    }

def update_fuzzy_config(**kwargs):
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
    config = get_fuzzy_config()
    print("="*50)
    print("Fuzzy Search Configuration:")
    print(f"  Enabled: {config['enabled']}")
    print(f"  Similarity Threshold: {config['threshold']}")
    print(f"  Max Candidates: {config['max_candidates']}")
    print(f"  Similarity Method: {config['similarity_method']}")
    print(f"  Query Normalization: {config['normalize_query']}")
    print(f"  Remove Punctuation: {config['remove_punctuation']}")
    print("="*50)