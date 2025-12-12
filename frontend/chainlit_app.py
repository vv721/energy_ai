import chainlit as cl
import os
import sys

# 正确设置项目根目录路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入配置
from config import DEFAULT_PROVIDER, DEFAULT_MODEL, TEMPERATURE, MAX_TOKENS, VECTORSTORE_PATH, COLLECTION_NAME
from utils import handle_exc

# 在类定义之前导入backend模块，确保路径正确
try:
    from backend.rag import VectorStoreManager, DocumentProcessor, RAGChain
    BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"❌ 导入backend模块失败: {e}")
    BACKEND_AVAILABLE = False
    VectorStoreManager = DocumentProcessor = RAGChain = None

# 直接创建RAGService类
class RAGService:
    def __init__(self):
        self.vector_store_manager = None
        self.docs_processor = None
        self.rag_chain = None
        self.backend_available = BACKEND_AVAILABLE
        self.vector_store_loaded = False

    def init_vector_store_manager(self):
        if not self.backend_available:
            handle_exc(Exception("backend模块不可用"), "初始化向量存储管理器失败")
            return None
            
        try:
            self.vector_store_manager = VectorStoreManager(persist_directory=VECTORSTORE_PATH)
            print("✅ 向量存储管理器初始化成功")
            return self.vector_store_manager
        except Exception as e:
            handle_exc(e, "初始化向量存储管理器失败")
            return None
        
    def load_vector_store(self):
        """加载向量存储集合"""
        if not self.backend_available:
            handle_exc(Exception("backend模块不可用"), "加载向量存储失败")
            return False
            
        if not self.vector_store_manager:
            self.init_vector_store_manager()

        try:
            result = self.vector_store_manager.load_vector_store(COLLECTION_NAME)
            if result:
                print("✅ 向量存储加载成功")
                self.vector_store_loaded = True
                return True
            else:
                print("⚠️ 向量存储不存在，需要先上传文档")
                return False
        except Exception as e:
            handle_exc(e, "加载向量存储失败")
            return False
        
    def init_docs_processor(self):
        if not self.backend_available:
            handle_exc(Exception("backend模块不可用"), "初始化文档处理器失败")
            return None
            
        try:
            self.docs_processor = DocumentProcessor()
            print("✅ 文档处理器初始化成功")
            return self.docs_processor
        except Exception as e:
            handle_exc(e, "初始化文档处理器失败")
            return None
        
    def init_rag_chain(self):
        if not self.backend_available:
            handle_exc(Exception("backend模块不可用"), "初始化RAG链失败")
            return None
            
        try:
            if not self.vector_store_manager:
                self.init_vector_store_manager()

            self.rag_chain = RAGChain(self.vector_store_manager)
            print("✅ RAG链初始化成功")
            return self.rag_chain
        except Exception as e:
            handle_exc(e, "初始化RAG链失败")
            return None
        
    def get_sample_docs(self, query: str = "test", k: int = 5):
        if not self.vector_store_manager:
            self.init_vector_store_manager()
        try:
            return self.vector_store_manager.similar_search(query, k)
        except Exception as e:
            handle_exc(e, "获取样本文档失败")
            return None
        
    def del_coll(self):
        if not self.vector_store_manager:
            self.init_vector_store_manager()

        try:
            from utils import safe_del_res, force_gc_coll
            #clear resources
            safe_del_res(self.vector_store_manager)
            force_gc_coll()

            result = self.vector_store_manager.del_collection(COLLECTION_NAME)
            force_gc_coll()

            return result
        except Exception as e:
            handle_exc(e, "删除集合失败")
            return {"success": False, "error": str(e)}
        
    def add_docs(self, file_paths):
        if not self.docs_processor:
            self.init_docs_processor()

        if not self.vector_store_manager:
            self.init_vector_store_manager()

        try:
            # 处理文档
            docs = []
            for file_path in file_paths:
                print(f"📖 正在加载文件：{os.path.basename(file_path)}")
                doc = self.docs_processor.load_document(file_path)
                if doc:
                    docs.extend(doc)
                    print(f"✅ 文件 {os.path.basename(file_path)} 加载成功，共 {len(doc)} 个文档片段")

            if not docs:
                return {"success": False, "error": "未能加载任何文档内容"}

            print(f"✂️ 正在分割文档，共 {len(docs)} 个文档片段")
            chunks = self.docs_processor.split_documents(docs)

            if not chunks:
                return {"success": False, "error": "文档分割后无有效内容"}

            print(f"📊 文档分割完成，共 {len(chunks)} 个文本块")

            # 根据向量存储状态选择操作
            if self.vector_store_loaded:
                # 向量存储已存在，添加文档
                print("📥 向量存储已存在，正在添加文档...")
                result = self.vector_store_manager.add_documents(chunks)
                if result.get("success"):
                    result["message"] = f"成功添加 {len(chunks)} 个文本块到现有知识库"
            else:
                # 向量存储不存在，创建新的向量存储
                print("🆕 创建新的向量存储...")
                try:
                    self.vector_store_manager.create_vector_store(chunks, collection_name=COLLECTION_NAME)
                    self.vector_store_loaded = True
                    result = {"success": True, "message": f"成功创建知识库并添加 {len(chunks)} 个文本块"}
                except Exception as e:
                    result = {"success": False, "error": f"创建向量存储失败：{e}"}

            if result.get("success"):
                print("✅ 文档添加成功，向量存储已更新")
            return result
        except Exception as e:
            handle_exc(e, "添加文档失败")
            return {"success": False, "error": str(e)}
        
    def answer_question(self, question: str, llm_provider: str = "openai",
                        model_name: str = None, temperature: float = 0.1,
                        max_tokens: int = 1024):
        if not self.backend_available:
            return {"answer": "系统初始化失败，backend模块不可用", "source_docs": []}
            
        if not self.rag_chain:
            self.init_rag_chain()

        # 先尝试加载向量存储
        if not self.vector_store_loaded:
            if not self.load_vector_store():
                return {"answer": "当前没有可用的文档库，请先上传文档文件。", "source_docs": []}

        try:
            #设置QA链
            self.rag_chain.setup_qa_chain(llm_provider=llm_provider,
                                          model_name=model_name,
                                          temperature=temperature,
                                          max_tokens=max_tokens)
            
            result = self.rag_chain.answer_question(question)
            return result
        except Exception as e:
            handle_exc(e, "回答问题失败")
            return {"answer": f"回答问题失败: {str(e)}", "source_docs": []}

# 初始化服务（单例模式）
rag_service = RAGService()



@cl.on_chat_start
async def start_chat():
    """聊天开始时初始化"""
    if not rag_service.backend_available:
        await cl.Message(
            content="⚠️ 系统初始化失败：backend模块无法导入，请检查项目结构。"
        ).send()
        return
    
    # 尝试加载向量存储
    vector_store_loaded = rag_service.load_vector_store()
    
    if vector_store_loaded:
        await cl.Message(
            content=f"✅ 欢迎使用能源AI助手！当前使用模型：{DEFAULT_MODEL}\n\n您可以输入问题开始对话，或者使用左侧的上传按钮添加文档。"
        ).send()
    else:
        await cl.Message(
            content=f"⚠️ 欢迎使用能源AI助手！当前使用模型：{DEFAULT_MODEL}\n\n当前没有可用的文档库，请先上传文档文件。\n\n请使用左侧的上传按钮添加文档。"
        ).send()

@cl.on_message
async def main(message: cl.Message):
    """处理用户消息"""
    try:
        print(f"🔍 收到用户消息：'{message.content}'")
        
        # 检查消息是否包含文件（原生文件上传按钮上传的文件）
        if message.elements:
            print("📤 检测到通过原生文件上传按钮上传的文件")
            file_paths = []
            file_names = []
            
            # 处理上传的文件
            for element in message.elements:
                if hasattr(element, 'path') and element.path:
                    file_paths.append(element.path)
                    file_names.append(element.name if hasattr(element, 'name') else os.path.basename(element.path))
                    print(f"📄 检测到文件：{element.name if hasattr(element, 'name') else os.path.basename(element.path)}")
            
            if file_paths:
                # 发送开始处理消息
                await cl.Message(content=f"📤 开始处理 {len(file_paths)} 个文件：{', '.join(file_names)}").send()
                
                # 发送处理进度消息
                await cl.Message(content="⏳ 正在加载和分割文档内容...").send()
                
                # 处理文档
                result = rag_service.add_docs(file_paths)
                
                print(f"📊 文件处理结果：{result}")
                
                if result.get("success"):
                    # 发送详细成功消息
                    success_message = f"✅ 文档上传成功！\n\n"
                    success_message += f"📄 已上传文件：{', '.join(file_names)}\n"
                    success_message += f"📊 处理结果：{result.get('message', '文档已成功添加到知识库')}\n\n"
                    success_message += "💬 现在可以开始提问了！"
                    await cl.Message(content=success_message).send()
                else:
                    # 发送详细错误消息
                    error_msg = result.get("error", "未知错误")
                    error_message = f"❌ 上传失败\n\n"
                    error_message += f"📄 尝试上传的文件：{', '.join(file_names)}\n"
                    error_message += f"⚠️ 错误信息：{error_msg}\n\n"
                    error_message += "🔧 请检查文件格式或稍后重试"
                    await cl.Message(content=error_message).send()
                
                return
        
        
        # 正常的RAG问答处理
        print("💬 处理RAG问答请求")
        
        # 发送处理中消息
        await cl.Message(content="🤔 正在思考您的问题...").send()
        
        result = rag_service.answer_question(
            question=message.content,
            llm_provider=DEFAULT_PROVIDER,
            model_name=DEFAULT_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )

        elements = []
        if result.get("source_docs"):
            elements.append(
                cl.Text(
                    name="来源文档",
                    content="\n\n".join([doc.page_content for doc in result.get("source_docs", [])])
                )
            )
        
        # 发送回复
        await cl.Message(
            content=result.get("answer", "抱歉，无法生成回答"),
            elements=elements if elements else None
        ).send()
        
    except Exception as e:
        handle_exc(e, "处理消息时出错")
        await cl.Message(content="❌ 处理消息时出现错误，请稍后重试").send()