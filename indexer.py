from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
import os.path
from dotenv import  load_dotenv

load_dotenv(".env")

# ========== 1. 验证 PDF 文件路径 ==========
pdf_path = r'LLM.pdf'
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"❌ PDF 文件不存在！请检查路径：{pdf_path}")

# ========== 2. 加载并分割 PDF ==========
# 加载 PDF（支持图片文字解析，需 rapidocr-onnxruntime 已安装）
pdf_loader = PyPDFLoader(pdf_path, extract_images=True)
# 文本分割（适配中文的合理参数）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 每个文本片段 500 字
    chunk_overlap=50,  # 片段重叠 50 字，保证语义连贯
    separators=["\n\n", "\n", "。", "！", "？", "，", "、"]  # 中文分割符
)
chunks = pdf_loader.load_and_split(text_splitter=text_splitter)
print(f"📄 成功解析 PDF，分割为 {len(chunks)} 个文本片段")

# ========== 3. 加载嵌入模型 ==========
# 使用通义千问的嵌入模型
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# ========== 4. 生成并保存 FAISS 向量库 ==========
vector_db = FAISS.from_documents(chunks, embeddings)
vector_db.save_local('LLM.faiss')

# ========== 5. 输出成功信息 ==========
print(f"✅ 向量库生成成功！")
print(f"📁 向量库保存路径：{os.path.abspath('LLM.faiss')}")
