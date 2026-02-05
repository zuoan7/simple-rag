# 从langchain社区库导入FAISS：轻量级本地向量数据库
# 作用：存储和检索已向量化的文档，这里用于加载提前构建好的向量库文件
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS

# 导入PromptTemplate：提示词模板构建工具
# 作用：定义固定的提示词格式，将「检索到的上下文」和「用户问题」自动填充到模板中，传递给大模型
from langchain.prompts import PromptTemplate  # 改用基础 PromptTemplate

# 导入RunnablePassthrough：链式执行中的数据透传工具
# 作用：在LangChain的链式流程中，传递数据而不做任何修改，这里用于构建对话链时透传用户查询
from langchain_core.runnables import RunnablePassthrough

# 导入itemgetter：用于从字典等可迭代对象中快速获取指定键的值
# 作用：在对话链中，从用户输入的字典中提取「query」（用户问题）字段，分别用于检索和填充提示词
from operator import itemgetter

# 导入os模块：用于处理系统路径、环境变量等（注：这段代码中未实际使用，属于冗余导入）
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(".env")


embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# ========== 加载 FAISS 向量库 ==========
# 从本地加载已提前构建好的FAISS向量数据库
vector_db = FAISS.load_local(
    'LLM.faiss',  # 本地向量库的文件夹路径（包含索引文件和数据文件）
    embeddings,  # 传入上面初始化的嵌入模型，用于解析向量库中的数据（必须和构建向量库时的模型一致）
    allow_dangerous_deserialization=True  # 允许危险的反序列化操作
    # 说明：FAISS本地加载时，会涉及数据反序列化，LangChain默认禁用该操作以保证安全
    # 作用：解除安全限制，成功加载本地向量库（仅用于本地测试，生产环境不建议开启）
)

# 将FAISS向量数据库转换为「检索器」（retriever）
# 作用：检索器封装了相似性搜索的逻辑，可直接接收用户问题，返回相关的文档片段
retriever = vector_db.as_retriever(
    search_kwargs={"k":5}  # 设置检索参数：k=5 表示返回和用户问题最相关的5条文档
)
# 说明：k值可调整，k值越大，检索到的上下文越多，但可能引入冗余信息，也会增加模型生成的压力

# ========== 调用 千问对话大模型 ==========
# 初始化Ollama模型实例，连接本地Ollama服务
llm = ChatOpenAI(
    model="qwen-turbo",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ========== 简化 Prompt 模板（适配 LLM 类） ==========
# 定义提示词模板字符串，这是RAG的核心之一，直接影响模型的回答质量
prompt_template = """
你是一个乐于助人的助手，仅基于给定的上下文回答问题，不要编造信息。

上下文：
{context}

问题：{query}

回答：
"""
# 模板说明：
# 1. 明确模型的角色和规则：「仅基于上下文回答，不编造信息」，避免模型幻觉
# 2. {context}：占位符，后续会填充从FAISS中检索到的相关文档
# 3. {query}：占位符，后续会填充用户输入的问题
# 4. 固定的格式让模型更容易理解任务要求，提升回答的准确性

# 将模板字符串转换为LangChain可使用的PromptTemplate实例
prompt = PromptTemplate(
    input_variables=["context", "query"],  # 定义模板中的占位符变量，必须和模板中的{context}、{query}对应
    template=prompt_template  # 传入上面定义的模板字符串
)
# 作用：后续链式执行时，会自动将「检索到的上下文」和「用户问题」填充到占位符中，生成完整的提示词

# ========== 对话链（适配 LLM 类） ==========
# 构建RAG对话链，这是LangChain的核心链式执行逻辑，按顺序执行各个步骤
chat_chain = (
    # 第一步：数据预处理，构建一个字典，包含后续需要的「context」和「query」
    {
        # 「context」（上下文）：从用户输入中提取「query」，传递给检索器，获取相关文档
        # itemgetter("query")：从输入字典中提取「query」字段（用户问题）
        # |：LangChain中的链式操作符，相当于将前一个步骤的输出作为后一个步骤的输入
        "context": itemgetter("query") | retriever,
        # 「query」（问题）：直接从输入字典中提取「query」，透传给后续步骤
        "query": itemgetter("query")
    }
    # 第二步：将预处理后的字典（包含context和query）传递给prompt，填充提示词模板，生成完整提示词
    | prompt
    # 第三步：将完整提示词传递给llm（Ollama模型），生成最终回答
    | llm
)
# 对话链执行流程总结：用户输入{query: 问题} → 提取query检索上下文 → 填充提示词模板 → 模型生成回答
# 作用：封装完整的RAG流程，后续只需调用chat_chain.invoke()即可执行整个流程，无需分步调用

# ========== 启动对话 ==========
# 打印启动提示信息，告知用户系统已就绪
print("✅ RAG 对话系统已启动！")
print("💡 输入和 PDF 相关的问题即可提问（输入空行退出）")

# 构建无限循环，实现持续对话
while True:
    # 接收用户输入的问题，存储在query变量中
    query = input('\n问题: ')
    # 判断用户输入是否为空（去除前后空格后）
    if not query.strip():
        # 若为空，打印退出提示，跳出循环，结束对话
        print("👋 退出对话")
        break
    # 异常处理：捕获执行过程中的错误（如模型未启动、向量库不存在等），避免程序直接崩溃
    try:
        # 调用对话链的invoke方法，执行RAG流程，传入用户输入的问题（字典格式，键必须为query，和对话链定义一致）
        response = chat_chain.invoke({"query": query})
        # 打印模型生成的回答
        print(f"回答：{response.content}")
    # 捕获所有异常，存储在e变量中
    except Exception as e:
        # 打印错误信息，只截取前300个字符，避免控制台输出过长
        print(f"❌ 出错了：{str(e)[:300]}")