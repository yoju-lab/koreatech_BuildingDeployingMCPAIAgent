# -*- coding: utf-8 -*-
"""
두 개의 MCP 서버 도구(math, weather)를 사용하는 LangGraph ReAct 에이전트 예제
- math: 로컬 STDIO FastMCP 서버 (예: servers/math_server.py)
- weather: Streamable HTTP FastMCP 서버 (OpenWeatherMap 연동, 예: http://127.0.0.1:8000/mcp)
필요 패키지:
    pip install langgraph langchain-openai langchain-mcp-adapters python-dotenv
환경:
    OPENAI_API_KEY (필수)
실행:
    python agents/react_mcp_agent.py
"""

import os
import inspect
from pathlib import Path
import asyncio
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# --- .env 로드: 스크립트 상위 폴더의 .env를 명시적으로 찾고, 기존 env를 덮어쓰기 ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

async def build_agent() -> tuple[object, MultiServerMCPClient]:
    """
    - 두 MCP 서버를 선언하고 도구를 수집해 ReAct 에이전트를 생성합니다.
    - math: stdio, weather: streamable_http
    """
    # 1) MCP 클라이언트(멀티 서버)
    client = MultiServerMCPClient({
        # 로컬 STDIO 수학 서버
        "math": {
            "command": "python",
            # 절대경로 권장: 수업 자료 구조에 맞게 수정하세요.
            "args": [os.path.abspath("servers/math_server.py")],
            "transport": "stdio",
        },
        # 원격 HTTP(OpenWeatherMap 사용) 날씨 서버
        "weather": {
            "url": "http://127.0.0.1:8000/mcp",   # weather_server.py가 노출하는 엔드포인트
            "transport": "streamable_http",
            # 인증 헤더가 필요하면 여기에 추가:
            # "headers": {"Authorization": "Bearer <TOKEN>"}
        },
    })

    # 2) MCP 서버들이 노출한 모든 "툴"을 LangChain Tool로 변환
    tools = await client.get_tools()

    # 3) LLM 생성
    #    LangChain 래퍼를 사용하지만, 내부적으로 OPENAI_API_KEY를 사용합니다.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 4) ReAct 프리빌트 에이전트 생성 (LangGraph)
    agent = create_react_agent(llm, tools)
    return agent, client


async def demo_queries(agent):
    """
    데모 쿼리 2개:
      A) 현재 날씨 + 수식 계산
      B) 3일 예보 요약(근거 포함) + 간단 계산
    """
    # A) 현재 날씨값으로 수식 계산
    msg_a = (
        "Seoul, KR의 현재 기온(섭씨)과 바람(m/s)을 weather.current로 가져오고, "
        "그 값을 이용해 (기온*2 + 바람/3)을 계산해 소수점 둘째 자리로 보고해줘. "
        "가능하면 근거(측정 시각/단위)도 덧붙여."
    )
    result_a = await agent.ainvoke({"messages": [{"role": "user", "content": msg_a}]})
    print("\n[A] === FINAL ANSWER ===\n", result_a["messages"][-1].content)


    # B) 3일 예보 요약 + 간단 계산(최고-최저 평균차)
    msg_b = (
        "Seoul, KR의 3일 예보를 weather.forecast로 조회해서 날짜별 요약을 만들고, "
        "각 날짜에 대해 (최고-최저)의 평균을 구해 소수점 둘째 자리로 알려줘. "
        "요약에는 근거 링크/출처도 포함해."
    )
    result_b = await agent.ainvoke({"messages": [{"role": "user", "content": msg_b}]})
    print("\n[B] === FINAL ANSWER ===\n", result_b["messages"][-1].content)


async def main():
    print(os.getenv("OPENAI_API_KEY"))
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("환경변수 OPENAI_API_KEY가 필요합니다.")

    agent, client = await build_agent()
    try:
        # 데모 쿼리 실행
        await demo_queries(agent)
    finally:
        # --- 버전별 종료 메서드 차이 대응 ---
        close_fn = getattr(client, "aclose", None) or getattr(client, "close", None) or getattr(client, "astop", None)
        if close_fn:
            maybe_coro = close_fn()
            if inspect.iscoroutine(maybe_coro):
                await maybe_coro


if __name__ == "__main__":
    asyncio.run(main())
