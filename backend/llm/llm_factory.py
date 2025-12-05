"""LLM 工厂：统一管理不同 LLM 提供者的接入。

当前实现：
- 优先在运行时使用 LangChain 的 `ChatOpenAI`（若安装并可用）。
- 回退到 OpenAI 官方 `openai` 包（≥1.0.0）的客户端模式调用。
- 支持通过环境变量或参数传入 `api_key` 与 `base_url`。
"""
#优化类型注解
from __future__ import annotations 
from typing import Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
import os



load_dotenv()

try:
    from langchain_openai import ChatOpenAI
    from langchain.messages import HumanMessage,SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from langchain_core.runnables import Runnable
except ImportError:
    from langchain_core.runnables.base import Runnable

ENERGY_SYSTEM_PROMPT = """你是一个专业的能源AI助手，专注于回答与能源相关的问题，包括但不限于：
- 能源生产（煤炭、石油、天然气、风电、光伏、水电等）
- 能源消耗与效率
- 可再生能源技术与应用
- 能源政策与法规
- 能源经济与市场
- 能源存储与电网

请针对用户的问题提供准确、专业的回答。如果问题与能源无关，请礼貌地说明你主要专注于能源领域的问题。"""

#抽象基类
class BaseLLM:
    def chat(self, prompt: str) -> str:
        raise NotImplementedError()
    
    def invoke(self, input: str | dict, config: Optional[dict] = None) -> str:
        """LangChain Runnable 接口方法"""
        if isinstance(input, dict) and "input" in input:
            return self.chat(input["input"])
        return self.chat(input)


class OpenAIPythonLLM(BaseLLM, Runnable):
    def __init__(
        self, 
        model_name: str, 
        api_key: str, 
        api_base: Optional[str],  # 对应 OpenAI SDK 的 base_url 参数
        temperature: float = 0.7, 
        max_tokens: int = 1000
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.api_base = api_base

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base  # 新版用 base_url，替代旧版的 api_base
        )

    def chat(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": ENERGY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
            ]
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content.strip()
    
    def invoke(self, input: str | dict) -> str:
        """LangChain Runnable 接口实现"""
        if isinstance(input, dict) and "input" in input:
            return self.chat(input["input"])
        return self.chat(str(input))


class LangChainLLM(BaseLLM, Runnable):
    def __init__(
        self, 
        model_name: str, 
        temperature: float = 0.7, 
        max_tokens: int = 1000,
        api_key: Optional[str] = None,  # 新增：传递 api_key
        api_base: Optional[str] = None  # 新增：传递 base_url
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = ChatOpenAI(
            model=self.model_name,  # 新版参数用 model 替代 model_name（兼容但推荐）
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=api_key,       # 显式传递 api_key
            base_url=api_base      # 显式传递 base_url（替代旧版的 openai_api_base）
        )

    def chat(self, prompt: str) -> str:
        messages = [
            SystemMessage(content=ENERGY_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        resp = self._client.invoke(messages)
        return resp.content.strip()
    
    def invoke(self, input: str | dict) -> str:
        """LangChain Runnable 接口实现"""
        if isinstance(input, dict) and "input" in input:
            return self.chat(input["input"])
        return self.chat(str(input))


def get_llm(
    provider: str = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> BaseLLM:
    """工厂方法，返回具体 LLM 实例。

    参数优先级：函数参数 -> 环境变量 -> 默认值。
    支持 OpenAI 和 Aliyun (DashScope) 两种 Provider。
    """
    # 规范化 provider（转小写处理）
    provider = (provider or os.getenv("DEFAULT_PROVIDER", "aliyun")).lower()
    
    # 根据 provider 选择默认模型
    if provider == "openai":
        model_name = model_name or os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    elif provider == "aliyun":
        model_name = model_name or os.getenv("MODEL_NAME", "qwen-turbo")
    else:
        model_name = model_name or os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    
    temperature = float(temperature) if temperature is not None else float(os.getenv("TEMPERATURE", 0.1))
    max_tokens = int(max_tokens) if max_tokens is not None else int(os.getenv("MAX_TOKENS", 1000))

    # 根据 provider 获取 API Key 和 Base URL
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    elif provider == "aliyun":
        api_key = os.getenv("ALIYUN_API_KEY")
        api_base = os.getenv("ALIYUN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    else:
        raise ValueError(f"不支持的 provider: {provider}。支持的选项: openai, aliyun")
    
    if not api_key or api_key == "your_key":
        raise ValueError(
            f"[{provider.upper()} 配置错误] API Key 未配置或为默认值。"
            f"请在 .env 文件中检查相关配置。"
        )

    # 优先使用 LangChain（若可用）
    if LANGCHAIN_AVAILABLE:
        try:
            # 【关键修改】传递 api_key 和 api_base 给 LangChainLLM
            return LangChainLLM(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
                api_base=api_base
            )
        except Exception as e:
            print(f"LangChain 调用失败，回退到原生 OpenAI：{e}")
            return OpenAIPythonLLM(
                model_name=model_name,
                api_key=api_key,
                api_base=api_base,
                temperature=temperature,
                max_tokens=max_tokens
            )
    else:
        return OpenAIPythonLLM(
            model_name=model_name,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens
        )


def test_connection() -> Tuple[bool, str]:
    try:
        llm = get_llm()
        out = llm.chat("请回复 短句: 连接成功")
        return True, out
    except Exception as e:
        return False, str(e)