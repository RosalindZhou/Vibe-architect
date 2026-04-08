# VibeArchitect Copilot

VibeArchitect Copilot 是一个基于 LangGraph 的溯源检索与多路自纠错系统架构 Copilot。利用先进的大语言模型技术（如 HyDE - Hypothetical Document Embeddings，和 RAG 检索增强生成）来理解用户的需求并自动起草、纠错和生成专业的大型系统架构。

## 特色功能

- **LangGraph 原生架构**：通过基于图的工作流与多路节点（包括假设性生成、检索、核心生成、自动打分裁判长等）实现自纠错与多次迭代循环，直到输出最优架构方案。
- **实时 Mermaid图表渲染**：不仅生成文本架构描述，还能通过内嵌在网页中的 Mermaid.js 可视化渲染生成的系统架构图。
- **RAG + 向量检索**：依靠 ChromaDB 与自定义检索策略为大语言模型提供实时的顶尖开源项目与架构片段知识源。
- **流式 UI 交互**：通过 Streamlit 实现现代化极致简约与流式输出体验，能够透明展示代理模型的每一步思考和状态。

## 项目结构

- `app.py`: Streamlit 前端交互与入口。
- `agent_core.py`: 基于 LangGraph 管理核心思维流与检索、生成、评估的节点定义。
- `data_pipeline.py`: 数据清洗、特征工程和模型准备模块。
- `evaluate.py`: 用于核心代码/系统的评估分析。
- `requirements.txt`: 运行本项目的环境依赖配置文件。

## 如何运行

1. 克隆本项目：
   ```bash
   git clone <repository_url>
   cd VibeArchitect
   ```

2. 配置虚拟环境并安装依赖：
   ```bash
   # 为避免依赖冲突，建议使用 venv 或 conda
   python -m venv .venv
   source .venv/bin/activate  # Windows 为: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. 环境变量与配置：
   请复制 `.env.example` 并重命名为 `.env`，填入所需的 API Keys 或其他机密配置。

4. 启动可视化界面：
   ```bash
   streamlit run app.py
   ```

## 开发与贡献

该架构还处于不断发展和完善的过程中。如果您对这种自纠错型 Agentic AI 机制感兴趣，欢迎 Fork 并提交 Pull Request。
