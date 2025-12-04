# Energy AI Assistant

轻量级能源公司用 AI 助手示例（基于 Python + LangChain + OpenAI）。

目标：
- 提供可扩展的 LLM 工厂，支持多个提供者（初始以 OpenAI 为主），并支持 `api_base`/`base_url` 配置。
- 提供一个简洁的 Streamlit 前端，允许选择模型、填写 base_url 与 API key（优先使用环境变量）。

快速开始（Windows PowerShell）：

```powershell
Set-Location 'd:\Codes\Proj_codes\energy_ai'
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
# 复制 .env.example 为 .env 并填充你的 OPENAI_API_KEY
copy .env.example .env
code .  # 打开 VS Code 并选择 .venv\Scripts\python.exe 作为解释器
# 后端快速连通性测试
python -m backend
# 启动前端 (Streamlit)
streamlit run frontend/app.py
```

项目结构摘要：

```
energy_ai/
  .vscode/
  backend/
    llm/llm_factory.py
    __main__.py
  frontend/app.py
  requirements.txt
  .env.example
  README.md
```

说明与扩展：
- `backend/llm/llm_factory.py` 为项目核心，封装了对 OpenAI 的直接调用并在可用时利用 LangChain 的 `ChatOpenAI`。
- 如果将来要添加其他 LLM 提供者（例如本地模型或厂商 SDK），在 `get_llm` 中添加新 provider 支持即可。
