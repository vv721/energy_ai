"""Backend entrypoint for quick connectivity test."""
from dotenv import load_dotenv
from backend.llm.llm_factory import test_connection


def main():
    load_dotenv()
    ok, msg = test_connection()
    if ok:
        print("LLM 连接成功：", msg)
    else:
        print("LLM 连接失败：", msg)


if __name__ == "__main__":
    main()