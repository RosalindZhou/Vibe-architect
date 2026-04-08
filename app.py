import streamlit as st
import re
import streamlit.components.v1 as components
from agent_core import VibeArchitectEngine

# 1. 页面配置与极简美学设计
st.set_page_config(
    page_title="VibeArchitect Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自定义 CSS 提升美感
st.markdown("""
<style>
    .stApp {
        background-color: #fafafa;
    }
    .stChatInputContainer {
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    h1 {
        font-family: 'Inter', sans-serif;
        background: -webkit-linear-gradient(45deg, #2a0845, #6441A5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
    }
    .mermaid {
        background: transparent;
        display: flex;
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

# 2. Mermaid 图表原生解析组件
def render_mermaid(diagram: str):
    """使用 streamlit.components.v1 和 mermaid.min.js 动态渲染 SVG"""
    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <script type="module">
          import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
          mermaid.initialize({{ startOnLoad: true, theme: 'default', securityLevel: 'loose' }});
        </script>
        <style>
          body {{ 
             display: flex; 
             justify-content: center; 
             align-items: center; 
             margin: 0; 
             background-color: transparent; 
             font-family: sans-serif;
          }}
        </style>
    </head>
    <body>
        <div class="mermaid">
        {diagram}
        </div>
    </body>
    </html>
    """
    components.html(html_code, height=550, scrolling=True)

def parse_and_render_markdown(content: str):
    """正则截取 Markdown 中的 mermaid 代码块并调用渲染"""
    pattern = r'```mermaid\n(.*?)```'
    matches = list(re.finditer(pattern, content, re.DOTALL))
    
    if not matches:
        st.markdown(content)
        return

    last_end = 0
    for match in matches:
        # 渲染前面的普通 Markdown
        st.markdown(content[last_end:match.start()])
        
        # 获取 Mermaid 源码并渲染图形
        mermaid_code = match.group(1).strip()
        render_mermaid(mermaid_code)
        
        # 保底提供一个可折叠的源码查看窗口
        with st.expander("🔍 展开查看 Mermaid 源码"):
            st.code(mermaid_code, language='mermaid')

        last_end = match.end()
        
    st.markdown(content[last_end:])

# 3. 状态初始化装配
if "engine" not in st.session_state:
    with st.spinner("正在加载 VibeArchitect 引擎与本地知识库..."):
        st.session_state.engine = VibeArchitectEngine()
    st.session_state.config = {"configurable": {"thread_id": "streamlit_user_session"}}
    st.session_state.messages = []

# 4. 前端 UI 结构
st.title("💡 VibeArchitect")
st.caption("基于 LangGraph 溯源检索与多路自纠错的系统架构 Copilot")

# 渲染历史对话
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("is_architect"):
            parse_and_render_markdown(msg["content"])
        else:
            st.markdown(msg["content"])

# 5. 聊天与状态栏流式交互
if prompt := st.chat_input("描述你想构建的系统，比如: '用 Next.js 和大模型做一个小说阅读器'"):
    
    # 追加并显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 显示 Agent 思考状态框
    with st.chat_message("assistant"):
        st_status = st.status("🤖 Agent 正在深度思考架构...", expanded=True)
        
        initial_state = {"user_input": prompt, "chat_history": [], "iteration_count": 0}
        
        try:
            # 流式捕获 LangGraph 节点事件
            for event in st.session_state.engine.graph.stream(initial_state, config=st.session_state.config):
                for node_name, state_update in event.items():
                    if node_name == "hyde_node":
                        st_status.write("🧠 **[HyDE]** 已生成假设性架构文本")
                    elif node_name == "retrieve_node":
                        st_status.write("📚 **[检索器]** 已匹配 5 个相关的 GitHub 顶级开源仓库片段")
                    elif node_name == "generate_node":
                        iteration = state_update.get('iteration_count', 1)
                        st_status.write(f"✍️ **[生成器]** 正在起草第 {iteration} 版系统架构图...")
                    elif node_name == "critic_node":
                        score = state_update.get("critic_score")
                        st_status.write(f"⚖️ **[裁判长]** 对当前草案严苛打分: **{score}/10**")
                        if score and score < 8:
                            st_status.write("🔄 分数未达标 (幻觉或技术栈有误)，强制触发自纠错循环！")
            
            st_status.update(label="架构设计完成！", state="complete", expanded=False)
            
            # 使用 get_state() 从 MemorySaver 中提取最终完美定稿
            final_state = st.session_state.engine.graph.get_state(st.session_state.config)
            final_arch = final_state.values.get("final_architecture", "生成架构失败。")
            
            # 存入列表记录并在网页上炫酷地渲染出来
            st.session_state.messages.append({"role": "assistant", "content": final_arch, "is_architect": True})
            parse_and_render_markdown(final_arch)
            
        except Exception as e:
            st_status.update(label="引擎发生致命错误", state="error", expanded=False)
            st.error(f"Error: {e}")
