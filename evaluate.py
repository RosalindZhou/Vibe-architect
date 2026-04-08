import os
import logging
import asyncio

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_groq import ChatGroq
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from agent_core import VibeArchitectEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Evaluate")

def main():
    logger.info("初始化 VibeArchitectEngine...")
    engine = VibeArchitectEngine()
    
    # 定义测试集
    test_queries = [
        "我想做一个人脸识别打卡系统，前端用 React，后端 Python",
        "我们要开发一个量化自动交易系统，使用大模型分析财经新闻的情绪并发出买卖信号"
    ]
    
    questions = []
    answers = []
    contexts_list = []
    
    config = {"configurable": {"thread_id": "eval_test_1"}}
    
    for i, query in enumerate(test_queries):
        logger.info(f"[{i+1}/{len(test_queries)}] 正在测试 Query: {query}")
        initial_state = {"user_input": query, "chat_history": [], "iteration_count": 0}
        
        # 使用 invoke 跑一遍完整的 Graph 流程
        final_state = engine.graph.invoke(initial_state, config=config)
        
        questions.append(query)
        answers.append(final_state.get("final_architecture", ""))
        contexts_list.append(final_state.get("retrieved_docs", []))
        
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
    })
    
    logger.info("开始使用 Ragas 跑评估指标 (Faithfulness / Answer Relevancy)...")
    
    # 包装用于裁判的大模型和 Embedding
    # Groq Llama 作为裁判模型，速度快零成本
    judge_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    wrapped_llm = LangchainLLMWrapper(judge_llm)
    
    base_embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    wrapped_embeddings = LangchainEmbeddingsWrapper(base_embeddings)
    
    # 运行全链路评估
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=wrapped_llm,
        embeddings=wrapped_embeddings
    )
    
    logger.info("====================================")
    logger.info("Ragas 量化评估结果出炉 🚀")
    logger.info("====================================")
    print(result)

if __name__ == "__main__":
    main()
