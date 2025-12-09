"""LLM 工厂：统一管理不同 LLM 提供者的接入"""
#优化类型注解
from __future__ import annotations 
from typing import Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
import os

from ..config import ENERGY_SYSTEM_PROMPT, get_llm_config
from ..exceptions import LLMConfigError, APIConnectionError

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


class BaseLLM:
    #抽象基类，通用LLM接口
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
        api_base: Optional[str],  
        temperature: float = 0.7, 
        max_tokens: int = 1000
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.api_base = api_base

        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_base 
            )
        except Exception as e:
            raise APIConnectionError(f"OpenAI API 连接失败: {e}")

    def chat(self, prompt: str) -> str:
        try:
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
        except Exception as e:
            raise APIConnectionError(f"OpenAI API 调用失败: {e}")
    
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
        api_key: Optional[str] = None,  
        api_base: Optional[str] = None  
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        try:
            self._client = ChatOpenAI(
                model=self.model_name,  # 新版参数用 model 替代 model_name（兼容但推荐）
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=api_key,       # 显式传递 api_key
                base_url=api_base      # 显式传递 base_url（替代旧版的 openai_api_base）
            )
        except Exception as e:
            raise APIConnectionError(f"LangChain OpenAI API 连接失败: {e}")

    def chat(self, prompt: str) -> str:
        try:
            messages = [
                SystemMessage(content=ENERGY_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ]
            resp = self._client.invoke(messages)
            return resp.content.strip()
        except Exception as e:
            raise APIConnectionError(f"LangChain OpenAI API 调用失败: {e}")
    
    def invoke(self, input: str | dict) -> str:
        """LangChain Runnable 接口实现"""
        if isinstance(input, dict) and "input" in input:
            return self.chat(input["input"])
        return self.chat(str(input))


class LLMFactory:
    #create and manager LLM instances
    @staticmethod
    def create_llm(
        provider: Optional["str"] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None
    ) -> BaseLLM:
        #返回具体LLM实例

        config = get_llm_config(provider)

        # 使用函数参数覆盖配置
        if model_name is not None:
            config["model_name"] = model_name
        if temperature is not None:
            config["temperature"] = temperature
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        if api_key is not None:
            config["api_key"] = api_key
        if api_base is not None:
            config["api_base"] = api_base

        if not config["api_key"] or config["api_key"]=="your_key":
            raise LLMConfigError(
                f"[{config['provider'].upper()} 配置错误] API Key 未配置或为默认值。"
                f"请在 .env 文件中检查相关配置。"
            )
    
        if LANGCHAIN_AVAILABLE:
            try:
                #优先使用langchain
                return LangChainLLM(
                    model_name=config["model_name"],
                    temperature=config["temperature"],
                    max_tokens=config["max_tokens"],
                    api_key=config["api_key"],
                    api_base=config["api_base"]
                )
            except Exception as e:
                print(f"LangChain 调用失败，回退到原生 OpenAI：{e}")
                return OpenAIPythonLLM(
                    model_name=config["model_name"],
                    api_key=config["api_key"],
                    api_base=config["api_base"],
                    temperature=config["temperature"],
                    max_tokens=config["max_tokens"]
                )
        else:
            return OpenAIPythonLLM(
                model_name=config["model_name"],
                api_key=config["api_key"],
                api_base=config["api_base"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"]
            )
    
    @staticmethod
    def test_connection() -> Tuple[bool, str]:
        try:
            llm = get_llm()
            out = llm.chat("请回复 短句: 连接成功")
            return True, out
        except Exception as e:
            return False, str(e)
        
def get_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
) -> BaseLLM:
    #向后兼容，内部调用create_llm
    return LLMFactory.create_llm(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        api_base=api_base
    )

def test_connection() -> Tuple[bool, str]:
    """向后兼容的函数，内部调用LLMFactory.test_connection"""
    return LLMFactory.test_connection()
