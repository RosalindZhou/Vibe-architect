import os
import time
import logging
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from github import Github, Auth
from github.GithubException import RateLimitExceededException, GithubException

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量定义
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# 搜索包含目标关键词的顶级仓库
REPO_SEARCH_QUERY = 'AI Agent OR RAG OR LLM API in:description,readme,name sort:stars-desc'
MAX_REPOS_TO_FETCH = 20
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 加载环境变量
load_dotenv()

class GithubDataPipeline:
    """
    负责从 GitHub 抓取数据并建立本地 Chroma 向量数据库以进行 RAG 检索的数据管道。
    """

    def __init__(self):
        """
        初始化数据管道：认证 GitHub、初始化嵌入模型和向量数据库路径。
        """
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise ValueError("未找到 GITHUB_TOKEN，请在 .env 文件中配置。")

        # 使用 Token 认证 GitHub
        auth = Auth.Token(github_token)
        self.g = Github(auth=auth)
        
        # 初始化 FastEmbed 嵌入模型 (本地运行，无需 API 密钥)
        self.embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
        
        # 记录初始化状态
        logger.info(f"GitHub API 及本地 {EMBEDDING_MODEL} 模型已成功初始化。")

    def wait_for_rate_limit(self) -> None:
        """
        处理 GitHub API 的速率限制 (Rate limit)。如果超限则自动休眠等待。
        """
        core_rate_limit = self.g.get_rate_limit().rate
        if core_rate_limit.remaining < 10:
            reset_timestamp = core_rate_limit.reset.timestamp()
            sleep_time = reset_timestamp - time.time() + 10 # 多加 10 秒缓冲
            if sleep_time > 0:
                logger.warning(f"达到 GitHub API 速率限制。休眠 {sleep_time:.0f} 秒直到重置...")
                time.sleep(sleep_time)

    def fetch_top_repositories(self, max_repos: int = MAX_REPOS_TO_FETCH) -> List[Any]:
        """
        根据搜索条件获取排名前列的仓库。
        
        Args:
            max_repos (int): 需要获取的最大仓库数量。
            
        Returns:
            List[Any]: PyGithub 仓库对象的列表。
        """
        logger.info(f"正在搜索仓库，查询条件: {REPO_SEARCH_QUERY}")
        repos = []
        try:
            self.wait_for_rate_limit()
            search_results = self.g.search_repositories(query=REPO_SEARCH_QUERY)
            
            for repo in search_results[:max_repos]:
                repos.append(repo)
                
            logger.info(f"成功获取到 {len(repos)} 个代码仓库。")
            return repos
        except Exception as e:
            logger.error(f"获取仓库失败: {e}")
            return repos

    def get_repo_files(self, repo: Any) -> Dict[str, str]:
        """
        获取指定仓库中的关键文件，主要是 README.md 和依赖文件。
        
        Args:
            repo (Any): PyGithub 仓库对象。
            
        Returns:
            Dict[str, str]: 文件内容，键为文件名（如 'README.md'），值为文件文本内容。
        """
        files_content = {}
        target_files = ["README.md", "README", "readme.md", "requirements.txt", "pyproject.toml"]
        
        for file_name in target_files:
            try:
                self.wait_for_rate_limit()
                # 尝试获取文件
                file_content = repo.get_contents(file_name)
                # 获取解码后的内容
                decoded_content = file_content.decoded_content.decode("utf-8", errors="ignore")
                files_content[file_name] = decoded_content
                logger.info(f"成功获取: {repo.full_name}/{file_name}")
            except GithubException as ge:
                # 404 文件未找到是正常现象
                if ge.status != 404:
                    logger.warning(f"尝试获取 {repo.full_name}/{file_name} 时发生错误: {ge.data}")
            except Exception as e:
                logger.warning(f"解析 {repo.full_name}/{file_name} 失败: {e}")
                
        return files_content

    def process_and_chunk_documents(self, repo_info: Any, files_content: Dict[str, str]) -> List[Document]:
        """
        处理和分块 Markdown 等文件内容。
        这部分逻辑非常关键：使用 MarkdownHeaderTextSplitter 基于标题进行语义分割，
        以保留架构和功能的上下文。备用方案采用字符级切割。
        
        Args:
            repo_info (Any): 仓库对象。
            files_content (Dict[str, str]): 包含文件内容的字典。
            
        Returns:
            List[Document]: LangChain Document 对象的列表。
        """
        documents = []
        
        # 1. 定义按 Markdown 标题分割的规则
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        # 2. 定义备用的按字符分割方案，用于处理极长的单个块 (Chunk)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )

        for file_name, content in files_content.items():
            base_metadata = {
                "repo_name": repo_info.full_name,
                "repo_url": repo_info.html_url,
                "stars": repo_info.stargazers_count,
                "file_name": file_name,
            }

            if file_name.lower().endswith('.md') or file_name.lower() == 'readme':
                # 首先执行基于语义（标题）的初级切分
                md_header_splits = markdown_splitter.split_text(content)
                
                # 由于头部切分后得到的块可能依旧很大，因此利用 Recursive Splitter 再过一遍
                final_splits = text_splitter.split_documents(md_header_splits)
                
                # 为所有分块注入仓库元数据
                for doc in final_splits:
                    doc.metadata.update(base_metadata)
                    documents.append(doc)
            else:
                # 对于依赖文件 (如 requirements.txt) 则直接进行按行/字符切割
                splits = text_splitter.split_text(content)
                for split in splits:
                    documents.append(Document(page_content=split, metadata=base_metadata))
                    
        return documents

    def build_vector_database(self, documents: List[Document]) -> None:
        """
        基于收集到的文档块构建本地 Chroma 向量数据库。
        
        Args:
            documents (List[Document]): 准备嵌入和存储的分块文档列表。
        """
        if not documents:
            logger.warning("没有可用于构建向量数据库的文档！")
            return
            
        logger.info(f"正在构建向量数据库，包含 {len(documents)} 个分块...")
        
        # 初始化或加载 Chroma，并将文档添加进去
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=CHROMA_DB_DIR
        )
        logger.info(f"向量数据库成功持久化至: {CHROMA_DB_DIR}")

    def run(self):
        """运行整个数据处理管道流程。"""
        logger.info("=== 启动 VibeArchitect 第一阶段: 数据管道 ===")
        os.makedirs(CHROMA_DB_DIR, exist_ok=True)
        
        all_documents = []
        repos = self.fetch_top_repositories()
        
        for repo in repos:
            logger.info(f"正在处理仓库: {repo.full_name} (Stars: {repo.stargazers_count})")
            
            files_content = self.get_repo_files(repo)
            if not files_content:
                logger.warning(f"仓库 {repo.full_name} 未获取到相关文件，跳过")
                continue
                
            docs = self.process_and_chunk_documents(repo, files_content)
            all_documents.extend(docs)
            logger.info(f"仓库 {repo.full_name} 已处理，提取了 {len(docs)} 个文档分块。")
            
        logger.info(f"=== 所有仓库处理完毕，准备进行 Embedding (总共 {len(all_documents)} 个条目) ===")
        self.build_vector_database(all_documents)
        logger.info("第一阶段数据管道顺利执行完毕！")

if __name__ == "__main__":
    try:
        pipeline = GithubDataPipeline()
        pipeline.run()
    except Exception as e:
        logger.error(f"发生致命错误: {e}")
