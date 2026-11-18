import os
import logging
from typing import List, Dict, Any, TypedDict
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from pprint import pprint
from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VibeArchitect.Agent")

# 常量设置
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "llama-3.3-70b-versatile"
MAX_ITERATIONS = 3

load_dotenv()

# ==========================================
# 1. State Definition
# ==========================================
class AgentState(TypedDict):
    """
    TypedDict representing the state of the multi-agent system.
    """
    user_input: str
    chat_history: List[str] # 简化版历史
    
    # 检索相关
    hypothetical_doc: str
    retrieved_docs: List[str]
    
    # 生成与迭代
    architecture_draft: str
    revision_feedback: str
    iteration_count: int
    critic_score: int
    
    # 最终输出
    final_architecture: str

# 结构化输出定义
class CriticOutput(BaseModel):
    """用于评估架构生成质量的打分系统"""
    score: int = Field(description="打分 1 到 10. 评分依据：幻觉程度 (不能编造知识库不存在的框架) 和相关性。8分及以上表示合格。")
    feedback: str = Field(description="具体的修改建议。如果满分则回复 '无修正建议'。")

# ==========================================
# 2. VibeArchitect Engine
# ==========================================
class VibeArchitectEngine:
    def __init__(self):
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("Set GROQ_API_KEY before starting VibeArchitect.")

        # 初始化强大的 Groq LLM
        self.llm = ChatGroq(model=LLM_MODEL, temperature=0.2)
        
        # 初始化结构化输出 Critic
        self.critic_llm = self.llm.with_structured_output(CriticOutput)

        # 初始化本地向量库与 Embedding
        logger.info("正在加载本地 Chroma 向量数据库进行基础检索...")
        self.embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = Chroma(
            persist_directory=CHROMA_DB_DIR, 
            embedding_function=self.embeddings
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        # 编译 LangGraph
        self.graph = self._build_graph()

    def _build_graph(self):
        """构建循环反馈图结构 (Cyclic LangGraph)"""
        workflow = StateGraph(AgentState)

        # 注册节点
        workflow.add_node("hyde_node", self.hyde_node)
        workflow.add_node("retrieve_node", self.retrieve_node)
        workflow.add_node("generate_node", self.generate_node)
        workflow.add_node("critic_node", self.critic_node)

        # 注册边
        workflow.set_entry_point("hyde_node")
        workflow.add_edge("hyde_node", "retrieve_node")
        workflow.add_edge("retrieve_node", "generate_node")
        workflow.add_edge("generate_node", "critic_node")
        
        # 条件路由
        workflow.add_conditional_edges(
            "critic_node",
            self.route_after_critic,
            {
                "generate_node": "generate_node",
                END: END
            }
        )

        # MemorySaver 用于记忆
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    # ---------------- 节点逻辑 ----------------

    def hyde_node(self, state: AgentState) -> Dict:
        """HyDE 假设性文档生成节点"""
        logger.info("=== Node [HyDE]: 正在生成假设性架构文档 ===")
        user_input = state["user_input"]
        
        # 让大模型假装写一篇关于该系统的 README 介绍
        system_prompt = """You are a senior AI system architect. The user will give you a rough idea. 
Please write a 1-paragraph hypothetical technical summary of how this system would work, mentioning standard open-source technologies you'd theoretcially use. 
Keep it dense, professional, and very similar to a GitHub repo description. Write the summary entirely in Simplified Chinese (简体中文)."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User Idea: {user_input}\n\nHypothetical Architecture Summary:")
        ]
        
        response = self.llm.invoke(messages)
        hypothetical_doc = response.content
        
        logger.info(f"-> HyDE 文档片段: {hypothetical_doc[:100]}...")
        return {"hypothetical_doc": hypothetical_doc, "iteration_count": 0}

    def retrieve_node(self, state: AgentState) -> Dict:
        """基础稠密向量检索节点"""
        logger.info("=== Node [Retrieve]: 正在通过 Chroma Vector 进行全库语义召回 ===")
        hypothetical_doc = state["hypothetical_doc"]
        
        # 1. 稠密检索 (Chroma Vector) -> Top 5
        final_top_docs = self.vector_store.similarity_search(hypothetical_doc, k=5)
        
        # 组装纯文本
        retrieved_texts = []
        for d in final_top_docs:
            repo = d.metadata.get('repo_name', 'Unknown Repo')
            content = d.page_content.replace('\n', ' ')
            retrieved_texts.append(f"[Repo: {repo}] {content}")
            
        logger.info(f"-> 成功召回出 {len(retrieved_texts)} 段历史源码片段。")
        return {"retrieved_docs": retrieved_texts}

    def generate_node(self, state: AgentState) -> Dict:
        """架构生成节点，支持基于 Critic 的反馈再修正"""
        iteration = state.get("iteration_count", 0)
        logger.info(f"=== Node [Generate]: 正在起草系统架构 (迭代轮次: {iteration + 1}) ===")
        
        user_input = state["user_input"]
        retrieved_docs = "\n".join(state["retrieved_docs"])
        feedback = state.get("revision_feedback", "")
        
        system_prompt = """You are 'VibeArchitect', an elite System Architecture Copilot.
Your job is to design a structured Markdown system architecture based on the User's Idea and the real Reference Context retrieved from top GitHub repos.
REQUIREMENTS:
1. Groundedness: Use ideas from the Reference Context! Do NOT invent fake open-source libraries.
2. Structure: Break it down into Frontend, Backend, Database/Vector DB, AI/LLM.
3. Visualization: You MUST include a Mermaid.js diagram block (```mermaid ... ```) visualizing the flow.
   - VERY IMPORTANT FOR MERMAID: 
     a. The very first line inside the ```mermaid block MUST be `flowchart TD` or `graph TD`.
     b. You MUST ONLY use alphanumeric characters (English) for Node IDs. Never use Chinese, spaces, or special characters for Node IDs!
     c. If your node labels contain spaces, brackets, parentheses or Chinese characters, you MUST wrap the label text in double quotes! Example: `id1["API Gateway (FastAPI)"]` or `id2["向量数据库 [Chroma]"]`. 
     d. Do NOT use HTML formatting inside labels.
4. Write the proposal, explanations, and diagram labels in English.
"""
        
        # 如果这是修改轮次，我们要加进来 Critic 的无情嘲讽/反馈
        if iteration > 0 and feedback:
            system_prompt += f"\n\nCRITIC FEEDBACK FROM PREVIOUS DRAFT (FIX THESE IMMEDIATELY!!):\n{feedback}"
            
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Reference Context:\n{retrieved_docs}\n\nUser Idea: {user_input}\n\nPlease generate the Architecture Markdown:")
        ]
        
        response = self.llm.invoke(messages)
        draft = response.content
        
        return {
            "architecture_draft": draft, 
            "iteration_count": iteration + 1
        }

    def critic_node(self, state: AgentState) -> Dict:
        """苛刻的架构评审节点"""
        logger.info("=== Node [Critic]: 正在进行幻觉与架构逻辑校验 ===")
        draft = state["architecture_draft"]
        retrieved_docs = "\n".join(state["retrieved_docs"])
        
        prompt = f"""You are a strict architecture reviewer.
Evaluate the following Architecture Draft based on the Reference Context.
1. Hallucination Check: Did they mention made-up libraries not supported by standard practice or the context?
2. Quality: Is there a valid Mermaid diagram? Are the technologies appropriate?
3. Mermaid Syntax Check: 
   - Does the diagram start with `flowchart TD` or `graph TD`?
   - Are all node IDs STRICTLY alphanumeric (no Chinese or spaces)?
   - Are all node labels containing spaces, brackets, etc. properly wrapped in double quotes?

Out of 10, give a score. If score < 8, provide specific feedback to fix.

Reference Context: {retrieved_docs}

Architecture Draft: {draft}
"""
        # invoke with structured output
        try:
            result: CriticOutput = self.critic_llm.invoke(prompt)
            score = result.score
            feedback = result.feedback
        except Exception as e:
            logger.warning(f"Critic 结构化输出解析失败，降级给满分处理: {e}")
            score = 10
            feedback = ""
            
        logger.info(f"-> Critic 评分: {score}/10 | 反馈: {feedback}")
        
        return {
            "critic_score": score,
            "revision_feedback": feedback,
            "final_architecture": draft # 最后一次的 Draft 作为候选 Final
        }

    def route_after_critic(self, state: AgentState) -> str:
        """决定是打回去重写，还是结束 Graph"""
        score = state["critic_score"]
        iteration = state["iteration_count"]
        
        if score >= 8:
            logger.info("成绩达标，架构被接受！")
            return END
        elif iteration >= MAX_ITERATIONS:
            logger.info("达到最大重试次数限制，强制输出当前最好版本。")
            return END
        else:
            logger.warning("成绩不合格，发回 Generator 进行自修正！")
            return "generate_node"

if __name__ == "__main__":
    print("\n\n" + "="*50)
    print("Welcome to VibeArchitect (Algorithmic RAG Edition)")
    print("="*50 + "\n")
    
    try:
        engine = VibeArchitectEngine()
        config = {"configurable": {"thread_id": "cli_test_1"}}
        
        print("[提示: 输入 'quit' 或 'exit' 退出]\n")
        while True:
            user_msg = input("描述你想做的系统 (e.g. '我想做一个帮公司提取财报数据的RAG系统'): ")
            if user_msg.lower() in ['quit', 'exit']:
                break
            if not user_msg.strip():
                continue
                
            print("\n" + "-"*40)
            initial_state = {"user_input": user_msg, "chat_history": [], "iteration_count": 0}
            
            # Streaming events from graph
            for event in engine.graph.stream(initial_state, config=config):
                for node_name, state_update in event.items():
                    pass # 输出都在 logger 里了
                    
            # 完整结束后提取最终架构
            final_state = engine.graph.get_state(config)
            print("\n" + "="*50)
            print("最终架构方案")
            print("="*50)
            print(final_state.values.get("final_architecture", "生成失败"))
            print("="*50 + "\n")
            
    except Exception as e:
        logger.error(f"引擎启动失败: {e}")
