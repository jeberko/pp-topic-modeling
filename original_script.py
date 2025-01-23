from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
from pathlib import Path
import numpy as np
from tqdm import tqdm

class CodeVectorizer:
    def __init__(self, max_df=1.0, min_df=1):
        self.max_df = max_df
        self.min_df = min_df
        # adding python and yaml for a test repo
        self.file_extensions = {'.cpp', '.hpp', '.h', '.cc', '.py'}
        
        self.semantic_groups = {
            # Core Data Structures
            'data_structures': {
                'vector', 'deque', 'map', 'list', 'array', 'queue', 'stack', 'tree', 'graph', 
                'set', 'hash_map', 'hash_set', 'priority_queue', 'heap', 'linked_list',
                'binary_tree', 'trie', 'matrix', 'tuple', 'pair'
            },
            
            # General Operations
            'operations': {
                'process', 'compute', 'calculate', 'transform', 'convert', 'modify', 'update',
                'validate', 'verify', 'normalize', 'parse', 'serialize', 'deserialize',
                'encode', 'decode', 'encrypt', 'decrypt', 'hash', 'merge', 'split', 'interpolate'
            },
            
            # Media Processing
            'media': {
                'image', 'video', 'frame', 'audio', 'picture', 'stream', 'codec', 'bitrate',
                'resolution', 'pixel', 'color', 'rgba', 'rgb', 'yuv', 'hsv', 'brightness',
                'contrast', 'saturation', 'hue', 'filter', 'blur', 'sharpen', 'compress',
                'decompress', 'transcode', 'render', 'capture', 'playback', 'timeline'
            },
            
            # Algorithms
            'algorithms': {
                'detect', 'encode', 'compress', 'filter', 'sort', 'search', 'match',
                'optimize', 'classify', 'cluster', 'segment', 'track', 'recognize',
                'predict', 'estimate', 'interpolate', 'extrapolate', 'convolve',
                'transform', 'analyze', 'merge', 'partition', 'traverse'
            },
            
            # Memory Management
            'memory': {
                'allocate', 'deallocate', 'pointer', 'reference', 'heap', 'stack',
                'memory_pool', 'garbage_collect', 'smart_pointer', 'weak_ptr',
                'shared_ptr', 'unique_ptr', 'buffer', 'cache', 'page', 'segment',
                'arena', 'malloc', 'free', 'new', 'delete'
            },
            
            # Threading and Concurrency
            'threading': {
                'thread', 'mutex', 'lock', 'atomic', 'concurrent', 'parallel',
                'semaphore', 'condition_variable', 'barrier', 'spinlock',
                'async', 'future', 'promise', 'task', 'thread_pool', 'coroutine',
                'scheduler', 'synchronize', 'critical_section', 'deadlock'
            },
            
            # Input/Output Operations
            'io': {
                'input', 'output', 'file', 'stream', 'buffer', 'socket', 'pipe',
                'channel', 'port', 'serial', 'network', 'tcp', 'udp', 'http',
                'websocket', 'stdin', 'stdout', 'stderr', 'filesystem', 'directory',
                'path', 'read', 'write', 'seek', 'flush', 'close'
            },
            
            # Graphics and Visualization
            'graphics': {
                'window', 'canvas', 'context', 'draw', 'paint', 'render',
                'viewport', 'scene', 'mesh', 'texture', 'shader', 'vertex',
                'fragment', 'geometry', 'animation', 'transform', 'projection',
                'camera', 'light', 'material', 'particle'
            },
            
            # Mathematics
            'math': {
                'matrix', 'vector', 'quaternion', 'complex', 'integral',
                'derivative', 'polynomial', 'exponential', 'logarithm',
                'trigonometric', 'sin', 'cos', 'tan', 'sqrt', 'pow',
                'random', 'normalize', 'interpolate', 'extrapolate'
            },
            
            # System Operations
            'system': {
                'process', 'thread', 'signal', 'interrupt', 'timer',
                'clock', 'timestamp', 'environment', 'registry', 'service',
                'driver', 'device', 'module', 'plugin', 'hook', 'event',
                'monitor', 'logging', 'debug', 'trace'
            },
            
            # Security
            'security': {
                'encrypt', 'decrypt', 'hash', 'sign', 'verify', 'authenticate',
                'authorize', 'credential', 'token', 'certificate', 'key',
                'password', 'salt', 'cipher', 'signature', 'checksum',
                'permission', 'policy', 'role', 'identity'
            }
    }

    def is_source_file(self, file_path):
        return Path(file_path).suffix.lower() in self.file_extensions

    def _clean_code(self, content, file_ext):
        # Remove comments
        content = re.sub(r'//.*?\n|/\*.*?\*/', '', content, flags=re.DOTALL)
        
        # Remove includes and preprocessor directives
        content = re.sub(r'#include.*?\n|#define.*?\n', '', content)
        
        # Remove string literals
        content = re.sub(r'".*?"', '', content)
        
        # Remove numbers but keep identifiers with numbers
        content = re.sub(r'\b\d+\b', '', content)
        
        # Remove common C++ keywords
        cpp_keywords = {
            'int', 'double', 'float', 'char', 'void', 'bool', 'long', 'short',
            'class', 'struct', 'template', 'typename',
            'public', 'private', 'protected',
            'const', 'static', 'virtual',
            'return', 'if', 'else', 'for', 'while',
            'std', 'string', 'vector', 'auto',
            'true', 'false', 'nullptr', 'this', 'new', 'delete',
            'try', 'catch', 'throw', 'namespace', 'using', 'typedef'
        }
        
        # Split into words and filter
        words = content.split()
        words = [w for w in words if w.lower() not in cpp_keywords]
        
        # Split camelCase and PascalCase
        split_words = []
        for word in words:
            splits = re.findall(r'[A-Z][a-z]*|[a-z]+|[A-Z]{2,}(?=[A-Z][a-z]|\d|\W|$)|\d+', word)
            split_words.extend(splits)
        
        # Group related terms
        processed_words = []
        for word in split_words:
            word = word.lower()
            found_group = False
            for group_name, group_terms in self.semantic_groups.items():
                if word in group_terms:
                    processed_words.append(group_name)
                    found_group = True
                    break
            if not found_group:
                processed_words.append(word)
        
        return ' '.join(processed_words)

    def process_with_context(self, tokens, window_size=3):
        """Process tokens with context window to capture related terms"""
        contexts = []
        for i in range(len(tokens)):
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)
            context = tokens[start:end]
            contexts.append(' '.join(context))
        return contexts

    def create_sparse_matrix(self, repo_path):
        documents = []
        file_paths = []
        code_structures = {}
        
        print("Reading files...")
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                file_path = os.path.join(root, file)
                if self.is_source_file(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Analyze code structure
                            code_structures[file_path] = analyze_code_structure(content)
                            
                            cleaned_content = self._clean_code(
                                content,
                                Path(file_path).suffix.lower()
                            )
                            if cleaned_content.strip():  # Only add if not empty
                                documents.append(cleaned_content)
                                file_paths.append(os.path.relpath(file_path, repo_path))
                    except (UnicodeDecodeError, IOError) as e:
                        print(f"Error reading {file_path}: {e}")
                        continue

        print(f"Processing {len(documents)} files...")
        
        self.vectorizer = TfidfVectorizer(
            max_df=self.max_df,
            min_df=self.min_df,
            stop_words='english',
            token_pattern=r'(?u)\b[a-zA-Z_][a-zA-Z0-9_]*\b',
            max_features=1000
        )

        doc_term_matrix = self.vectorizer.fit_transform(documents)
        return doc_term_matrix, self.vectorizer.get_feature_names_out(), file_paths, code_structures

def analyze_code_structure(content):
    """Analyze code structure for additional context"""
    structure = {
        'classes': re.findall(r'class\s+(\w+)', content),
        'functions': re.findall(r'(?:void|int|double|bool)\s+(\w+)\s*\(', content),
        'includes': re.findall(r'#include\s*[<"]([^>"]+)[>"]', content),
        'imports': re.findall(r'(?:from|import)\s+([\w.]+)', content),
        'decorators': re.findall(r'@(\w+)', content)
    }
    return structure

def cluster_related_terms(term_matrix, feature_names, n_clusters=5):
    """Cluster related terms based on co-occurrence"""
    similarity_matrix = cosine_similarity(term_matrix.T)
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = clustering.fit_predict(similarity_matrix)
    
    term_clusters = {}
    for term, cluster in zip(feature_names, clusters):
        if cluster not in term_clusters:
            term_clusters[cluster] = []
        term_clusters[cluster].append(term)
    
    return term_clusters

def create_topic_summary(model, feature_names, doc_topics, files, code_structures, term_clusters):
    """Create comprehensive summary for LLM interpretation"""
    summaries = []
    
    # Get topic-word distributions
    topic_word_dist = []
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[:-10:-1]
        top_words = feature_names[top_words_idx]
        top_probs = topic[top_words_idx]
        
        term_pairs = []
        for word, prob in zip(top_words, top_probs):
            if prob > 0.1:
                term_pairs.append(f"{word} ({prob:.2f})")
        
        topic_word_dist.append(term_pairs)
    
    # Get document-topic assignments
    doc_assignments = []
    for doc_idx, (doc_topic_dist, file_path) in enumerate(zip(doc_topics, files)):
        primary_topic = doc_topic_dist.argmax()
        strength = doc_topic_dist[primary_topic]
        if strength > 0.3:
            doc_assignments.append({
                'file': file_path,
                'topic': primary_topic,
                'strength': strength
            })
    
    # Create LLM-friendly summary
    llm_prompt = f"""
    Code Analysis Summary:
    
    Topic Word Distributions:
    {'-' * 40}
    {'\n'.join([f'Topic {i+1}: {", ".join(terms)}' for i, terms in enumerate(topic_word_dist)])}
    
    Document Assignments:
    {'-' * 40}
    {'\n'.join([f'{d["file"]} → Topic {d["topic"]+1} ({d["strength"]:.2f})' for d in doc_assignments])}
    
    Code Structure Analysis:
    {'-' * 40}
    """
    
    for file_path, structure in code_structures.items():
        llm_prompt += f"\nFile: {file_path.split('/')[-1]}\n"
        for category, items in structure.items():
            if items:
                llm_prompt += f"{category.capitalize()}: {', '.join(items)}\n"
    
    llm_prompt += f"""
    Term Clusters:
    {'-' * 40}
    """
    
    for cluster_id, terms in term_clusters.items():
        llm_prompt += f"\nCluster {cluster_id + 1}: {', '.join(terms)}"
    
    llm_prompt += """
    
    Please provide:
    1. The main functionality of each topic
    2. The relationships between topics
    3. The overall purpose of this codebase
    4. Any potential missing or underrepresented topics
    5. Suggested refactoring or organization improvements
    """
    
    return llm_prompt

def analyze_matrix_stats(matrix, terms, files):
    """Print statistics about the sparse matrix"""
    print(f"\nMatrix shape: {matrix.shape}")
    print(f"Number of non-zero elements: {matrix.nnz}")
    print(f"Sparsity: {matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4%}")
    print(f"Memory usage: {matrix.data.nbytes / 1024 / 1024:.2f} MB")

def create_topic_model(doc_term_matrix, n_topics=5):
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        n_jobs=-1,
        learning_method='batch',
        max_iter=50,
        doc_topic_prior=0.1,
        topic_word_prior=0.01
    )
    doc_topics = lda.fit_transform(doc_term_matrix)
    return lda, doc_topics

def print_topics(model, feature_names, n_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[:-n_top_words-1:-1]
        top_words = feature_names[top_words_idx]
        top_probs = topic[top_words_idx]
        
        print(f"\nTopic {topic_idx + 1}:")
        for word, prob in zip(top_words, top_probs):
            print(f"  {word}: {prob:.4f}")

if __name__ == "__main__":
    # REPO_PATH = "./test_repo"
    REPO_PATH = "/Users/claytongibb/Documents/repositories/brandbrief"
    N_TOPICS = 10
    
    # Create vectorizer with appropriate parameters
    vectorizer = CodeVectorizer(max_df=1.0, min_df=1)
    
    # Create sparse matrix
    doc_term_matrix, terms, files, code_structures = vectorizer.create_sparse_matrix(REPO_PATH)
    
    # Cluster related terms
    term_clusters = cluster_related_terms(doc_term_matrix, terms)
    
    # Print corpus statistics
    print(f"\nNumber of documents: {doc_term_matrix.shape[0]}")
    print(f"Vocabulary size: {doc_term_matrix.shape[1]}")
    
    # Analyze matrix statistics
    analyze_matrix_stats(doc_term_matrix, terms, files)
    
    # Create and train topic model
    lda_model, doc_topics = create_topic_model(doc_term_matrix, N_TOPICS)
    
    # Print topics
    print("\nTop words in each topic:")
    print_topics(lda_model, terms)
    
    # Base prompt for LLM
    base_prompt = """
                    You are a software engineer/architect evaluating a codebase repository. All you have for your analysis is the output of a topic modeling model which provides the following information. You must due your best based on your expert knowledge to ascertain what the purpose and high level functionality of the codebase is.

                    Please provide:
                    • Very high level bullets of the main functionality
                    • Detailed explanation of each identified functionality
                    • Potential use cases for this codebase
                    • Technical architecture insights
                    • Suggestions for potential improvements or missing components

                    Analysis Input:
                    --------------
                    """
    
    # Create comprehensive summary for LLM
    llm_prompt = create_topic_summary(lda_model, terms, doc_topics, files, code_structures, term_clusters)
    
    # Combine prompts
    final_prompt = base_prompt + llm_prompt
    
    # Ensure the file is overwritten
    output_file = 'prompt.txt'
    try:
        # Remove the file if it exists
        if os.path.exists(output_file):
            os.remove(output_file)
        
        # Write new content
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_prompt)
        
        print(f"\nPrompt has been written to '{output_file}'")
        
    except Exception as e:
        print(f"Error writing to file: {e}")
        
    # Optionally print to console as well
    print("\nGenerated Prompt:")
    print("-" * 80)
    print(final_prompt)
    print("-" * 80)