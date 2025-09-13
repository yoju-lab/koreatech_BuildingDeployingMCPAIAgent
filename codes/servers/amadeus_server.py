# servers/amadeus_server.py
"""
FastMCP 기반 MCP 서버 (HTTP)
Amadeus 항공권 검색 Tool 제공

구성:
    - Tool: amadeus_search (항공권 후보 조회)
환경변수(.env):
    AMADEUS_CLIENT_ID=<your_client_id>
    AMADEUS_CLIENT_SECRET=<your_client_secret>

실행 (cmd / conda 환경 mcp_dev):
    conda activate mcp_dev
    python servers/amadeus_server.py
    → http://127.0.0.1:8010/mcp  (transport=streamable_http)

클라이언트(에이전트) 측 연결 예:
    MultiServerMCPClient({
        "amadeus": {
            "url": "http://127.0.0.1:8010/mcp",
            "transport": "streamable_http",
        }
    })

참고:
    - Amadeus Python SDK 동기 API 사용 (단순성 목적)
    - 네트워크/인증 오류 발생 시 {"error": "..."} 형태로 반환
"""
from fastmcp import FastMCP
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from amadeus import Client, ResponseError
import logging

# ---------------------------------------------------------------------------
# .env 로드: 프로젝트 루트(.env) 명시적 지정
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# MCP 서버 인스턴스 이름(도메인)을 "flight_search" 로 지정
mcp = FastMCP("flight_search")

# 로깅 (서버 전용). DEBUG 시 외부 noisy 로거 억제
SERVER_LOG_LEVEL = os.getenv("AMADEUS_SERVER_LOG_LEVEL", os.getenv("FLIGHT_AGENT_LOG_LEVEL", "INFO")).upper()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
server_logger = logging.getLogger("amadeus_server")
server_logger.setLevel(getattr(logging, SERVER_LOG_LEVEL, logging.INFO))
if SERVER_LOG_LEVEL == "DEBUG":
    for noisy in ["httpx", "httpcore", "amadeus", "fastmcp", "mcp", "mcp.client.streamable_http", "asyncio"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# 내부 헬퍼: 인증 키 로딩 & Client 생성
# ---------------------------------------------------------------------------

def _require_client() -> Client:
    """Amadeus Client 생성 (환경변수 검증 포함).

    Raises:
        RuntimeError: 필수 환경변수 누락 시
    """
    cid = os.getenv("AMADEUS_CLIENT_ID")
    csec = os.getenv("AMADEUS_CLIENT_SECRET")
    if not cid or not csec:
        raise RuntimeError("AMADEUS_CLIENT_ID, AMADEUS_CLIENT_SECRET 환경변수 필요")
    # Client 인스턴스는 stateless 하므로 매 호출마다 생성해도 무방 (간단성 우선)
    return Client(client_id=cid, client_secret=csec)

# ---------------------------------------------------------------------------
# Tool 정의: amadeus_search
#   - Amadeus Flight Offers Search (v2) 호출
#   - 단순 패스스루 형태로 응답 JSON 중 data 리스트를 offers 키로 래핑
#   - 필드 설명은 Amadeus 공식 문서 참고
# ---------------------------------------------------------------------------

@mcp.tool
def amadeus_search(
    origin: str,                 # 출발 공항 IATA (예: ICN)
    destination: str,            # 도착 공항 IATA (예: JFK)
    departureDate: str,          # 출발일 (YYYY-MM-DD)
    returnDate: Optional[str] = None,  # (왕복 시) 귀국일
    adults: int = 1,             # 성인 수
    children: int = 0,           # 소아 수
    infants: int = 0,            # 유아 수
    travelClass: Optional[str] = None, # ECONOMY | PREMIUM_ECONOMY | BUSINESS | FIRST
    nonStop: Optional[bool] = None,    # 직항 여부 (True/False)
    currencyCode: Optional[str] = None, # 통화 (예: KRW, USD)
    includedCheckedBagsOnly: Optional[bool] = None, # 무료 위탁수하물 포함 여부
    airlineFilter: Optional[List[str]] = None,      # 특정 항공사 IATA 코드 리스트
    max: int = 20               # 검색 결과 상한 (Amadeus 제한 내)
) -> Dict[str, Any]:
    """Amadeus 항공권 후보 조회 Tool.

    Parameters
    ----------
    origin, destination : str
        IATA 3-letter 공항 코드
    departureDate, returnDate : str
        날짜 문자열 (YYYY-MM-DD)
    adults, children, infants : int
        탑승객 인원 설정
    travelClass : str
        선호 클래스 (미지정 시 전체)
    nonStop : bool
        직항만 필터링
    currencyCode : str
        요금 통화 코드 (예: KRW)
    includedCheckedBagsOnly : bool
        위탁수하물 포함 운임만
    airlineFilter : List[str]
        제한할 항공사 코드 목록 (예: ["KE", "OZ"]) -> Amadeus 파라미터 includedAirlineCodes
    max : int
        반환 최대 오퍼 수

    Returns
    -------
    dict
        {"offers": [...]} 또는 {"error": "메시지"}
    """
    try:
        amadeus = _require_client()
    except Exception as e:
        server_logger.debug("AuthError: %s", e)
        return {"error": f"AuthError: {e}"}

    # Amadeus API 파라미터 매핑
    params: Dict[str, Any] = {
        "originLocationCode": origin.strip().upper(),
        "destinationLocationCode": destination.strip().upper(),
        "departureDate": departureDate,
        "adults": adults,
        "children": children,
        "infants": infants,
        "max": max,
    }
    if returnDate:
        params["returnDate"] = returnDate
    if travelClass:
        params["travelClass"] = travelClass
    if nonStop is not None:
        params["nonStop"] = nonStop
    if currencyCode:
        params["currencyCode"] = currencyCode
    if includedCheckedBagsOnly is not None:
        params["includedCheckedBagsOnly"] = includedCheckedBagsOnly
    if airlineFilter:
        params["includedAirlineCodes"] = ",".join(code.strip().upper() for code in airlineFilter if code)

    try:
        server_logger.debug("amadeus_search params=%s", params)
        response = amadeus.shopping.flight_offers_search.get(**params)
        server_logger.debug("amadeus_search success offers=%d", len(getattr(response, "data", []) or []))
        return {"offers": response.data}
    except ResponseError as e:  # SDK가 던지는 표준 오류
        # e.response.body 또는 e.response.result 에 상세 JSON 이 있을 수 있음
        try:
            detail = getattr(e, "response", None)
            if detail and hasattr(detail, "result"):
                server_logger.debug("ResponseError detail.result=%s", detail.result)
                return {"error": detail.result}
        except Exception:
            pass
        server_logger.debug("ResponseError str=%s", e)
        return {"error": str(e)}
    except Exception as e:
        server_logger.debug("UnhandledError: %s", e)
        return {"error": f"UnhandledError: {e}"}

# ---------------------------------------------------------------------------
# 엔트리 포인트
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Streamable HTTP 로 기동 (기존 weather_server와 동일 패턴)
    # 에이전트에서 http://127.0.0.1:8010/mcp 로 연결
    mcp.run(transport="http", host="127.0.0.1", port=8010, path="/mcp")
