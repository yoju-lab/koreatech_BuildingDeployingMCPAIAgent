# agents/flight_search_agent.py
# -*- coding: utf-8 -*-
"""
멀티 에이전트 기반 최저가 항공권 검색 (LangGraph + MCP)

구성 노드:
    1) Planner   : 사용자 입력 정규화 → search_params
    2) Shopper   : MCP Tool(amadeus_search) 호출 → offers
    3) Presenter : Top-K 정렬/표 생성 → result_table_pretty

확장 포인트(추후):
    - Deduper / Normalizer : 동일 운임 / 경유 편수 정규화
    - Price Verifier       : 가격 변동/재확인 Tool 추가 (amadeus_price_confirm 등)
    - LLM Planner          : 자연어 질의 → (origin, destination, date ...) 추출

실행 (cmd / conda env mcp_dev):
    conda activate mcp_dev
    python agents/flight_search_agent.py

사전 준비:
    1) Amadeus MCP 서버 기동
        python servers/amadeus_server.py
    2) .env 에 AMADEUS_CLIENT_ID / AMADEUS_CLIENT_SECRET 설정

입력 예:
    origin=ICN, destination=JFK, departureDate=2025-10-01, adults=1

출력:
    콘솔에 Top-10 항공권 표 (편명 / 출발지 / 출발시각 / 도착지 / 도착시각 / 등급 / 운임)
"""
import os
import asyncio
import json
import logging
from typing import TypedDict, List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from time import perf_counter

# LangGraph
from langgraph.graph import StateGraph, START
# MCP 클라이언트 (여러 서버를 동시에 다룰 수 있는 어댑터)
from langchain_mcp_adapters.client import MultiServerMCPClient

# ---------------------------------------------------------------------------
# 환경 로드 (.env)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

# ---------------------------------------------------------------------------
# 로깅 설정 (DEBUG 시 이 모듈만 상세, 외부 라이브러리는 WARNING으로 눌러서 노이즈 제거)
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("FLIGHT_AGENT_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("flight_search_agent")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
if LOG_LEVEL == "DEBUG":
    for noisy in ["httpx", "httpcore", "asyncio", "mcp", "mcp.client.streamable_http", "langchain", "urllib3"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# MultiServer MCP 클라이언트 준비
#   - amadeus_server.py 가 HTTP(127.0.0.1:8010) 로 기동되어 있어야 함
# ---------------------------------------------------------------------------
MCP_CLIENT = MultiServerMCPClient({
    "amadeus": {
        "url": "http://127.0.0.1:8010/mcp",
        "transport": "streamable_http",
    }
})

# ---------------------------------------------------------------------------
# 단계 진행 실시간 표시 유틸
# ---------------------------------------------------------------------------
_AGENT_START_TS = perf_counter()

def _elapsed() -> str:
    return f"{perf_counter() - _AGENT_START_TS:7.3f}s"

def print_phase(phase: str, event: str, detail: str = ""):
    """터미널에 현재 단계(phase)의 이벤트(START/DONE/ERROR)를 실시간 출력.

    형식: [경과시간] [PHASE] [EVENT] detail
    예  : [  0.512s] [PLANNING] [START] 사용자 입력 정규화
    """
    msg = f"[{_elapsed()}] [{phase}] [{event}]"
    if detail:
        msg += f" {detail}"
    print(msg, flush=True)

async def get_tools() -> Dict[str, Any]:
    """서버 Tool 리스트를 {name: tool} 형태로 반환.

    NOTE: 필요 시 장기 실행에서 캐싱 or lazy-load 로 교체 가능.
    """
    tool_list = await MCP_CLIENT.get_tools()
    return {t.name: t for t in tool_list}

# ---------------------------------------------------------------------------
# LangGraph State 정의
# ---------------------------------------------------------------------------
class FlightSearchState(TypedDict, total=False):
    user_input: Dict[str, Any]
    search_params: Dict[str, Any]
    offers: List[Dict[str, Any]]
    result_table: List[Dict[str, Any]]
    result_table_pretty: str
    error: str

# ---------------------------------------------------------------------------
# 유틸: 샘플 오퍼 (오류/미인증 환경 Fallback)
# ---------------------------------------------------------------------------

def _sample_offers() -> List[Dict[str, Any]]:
    return [
        {
            "itineraries": [
                {"segments": [
                    {
                        "carrierCode": "KE",
                        "number": "081",
                        "departure": {"iataCode": "ICN", "at": "2025-10-12T10:00:00"},
                        "arrival": {"iataCode": "JFK", "at": "2025-10-12T15:00:00"},
                    }
                ]}
            ],
            "price": {"grandTotal": "1234.56"},
            "travelerPricings": [
                {"fareDetailsBySegment": [
                    {"cabin": "ECONOMY", "includedCheckedBags": {"quantity": 1}}
                ]}
            ],
        }
    ]

# ---------------------------------------------------------------------------
# Node Factories
# ---------------------------------------------------------------------------

def make_planner_node():
    """Planner: 사용자 입력(user_input) → 검색 파라미터(search_params).

    - 현재는 값 검증과 기본값 주입만 수행.
    - LLM 자연어 파싱 필요 시 여기 확장.
    """
    def planner_node(state: FlightSearchState) -> FlightSearchState:
        print_phase("PLANNING", "START", "사용자 입력 정규화")
        user_input = state.get("user_input")
        if not user_input:
            from datetime import date, timedelta
            logger.warning("user_input 미지정 → 기본값 사용(ICN→JFK, +30d)")
            user_input = {
                "origin": "ICN",
                "destination": "JFK",
                "departureDate": (date.today() + timedelta(days=30)).isoformat(),
                "adults": 1,
            }

        # 간단 정규화
        for key in ("origin", "destination"):
            val = user_input.get(key)
            if isinstance(val, str):
                user_input[key] = val.strip().upper()[:3]
        if "adults" in user_input:
            try:
                user_input["adults"] = int(user_input["adults"])
            except Exception:
                user_input["adults"] = 1

        new_state = dict(state)
        new_state["search_params"] = user_input  # type: ignore[assignment]
        print_phase("PLANNING", "DONE", f"origin={user_input.get('origin')} destination={user_input.get('destination')}")
        return new_state
    return planner_node


def make_shopper_node(tools: Dict[str, Any]):
    """Shopper: amadeus_search Tool 호출 → offers 수집.

    호출 전략:
        1) tool.ainvoke(dict) 우선 시도
        2) 실패 시 **kwargs 형태 fallback
    에러 시:
        - error 필드 설정 & 샘플 데이터(FALLBACK) 사용 (환경: FLIGHT_AGENT_USE_SAMPLE=1)
    """
    async def shopper_node(state: FlightSearchState) -> FlightSearchState:
        print_phase("SHOPPING", "START", "Amadeus 항공권 조회")
        tool = tools.get("amadeus_search")
        if not tool:
            logger.error("amadeus_search Tool 미발견")
            new_state = dict(state)
            new_state["offers"] = []
            new_state["error"] = "amadeus_search tool not found"
            print_phase("SHOPPING", "ERROR", "Tool not found")
            return new_state

        params = state.get("search_params", {})
        result: Optional[Any] = None

        async def _invoke_once() -> Optional[Any]:
            try:
                if hasattr(tool, "ainvoke"):
                    return await tool.ainvoke(params)
                return tool.invoke(params)  # type: ignore[attr-defined]
            except TypeError:
                # kwargs fallback
                if hasattr(tool, "ainvoke"):
                    return await tool.ainvoke(**params)  # type: ignore[arg-type]
                return tool.invoke(**params)  # type: ignore[attr-defined,arg-type]
            except Exception as e:  # noqa
                logger.error("shopper_node 호출 실패: %s", e)
                raise

        def _extract_offers(res: Any) -> List[Dict[str, Any]]:
            if res is None:
                return []
            # 문자열 -> JSON
            if isinstance(res, str):
                try:
                    res = json.loads(res)
                except Exception as e:  # noqa
                    logger.debug("JSON 파싱 실패(무시): %s", e)
                    return []
            if not isinstance(res, dict):
                return []
            # 에러 응답 즉시 처리
            if res.get("error") and not res.get("offers"):
                return []
            offers_local = res.get("offers")
            if offers_local is None and isinstance(res.get("data"), list):
                offers_local = res.get("data")
            if (not offers_local) and isinstance(res.get("structuredContent"), dict):
                sc = res.get("structuredContent")
                if isinstance(sc, dict) and isinstance(sc.get("offers"), list):
                    offers_local = sc.get("offers")
            if (not offers_local) and isinstance(res.get("content"), list):
                for blk in res.get("content", []):
                    if not isinstance(blk, dict) or blk.get("type") != "text":
                        continue
                    txt = blk.get("text")
                    if not isinstance(txt, str):
                        continue
                    try:
                        parsed = json.loads(txt)
                        if isinstance(parsed, dict) and isinstance(parsed.get("offers"), list):
                            offers_local = parsed.get("offers")
                            break
                    except Exception:
                        continue
            if not isinstance(offers_local, list):
                return []
            return offers_local  # type: ignore[return-value]

        # 최대 5회 (최대 ~5초) 재시도: 스트리밍 완료 대기 목적
        offers: List[Dict[str, Any]] = []
        for attempt in range(5):
            try:
                result = await _invoke_once()
            except Exception as e:
                if attempt == 0:
                    # 즉시 실패 -> fallback 가능 여부 판단
                    new_state = dict(state)
                    new_state["error"] = str(e)
                    if os.getenv("FLIGHT_AGENT_USE_SAMPLE", "0") in ("1", "true", "True"):
                        new_state["offers"] = _sample_offers()
                    else:
                        new_state["offers"] = []
                    return new_state
                # 이후 재시도 실패는 계속 진행
            offers = _extract_offers(result)
            logger.debug("shopper_node attempt=%d offers_count=%d", attempt + 1, len(offers))
            if offers:
                break
            # 첫 응답에 아직 콘텐츠가 비어있다면 잠시 대기 후 재호출 (스트리밍 지연 가정)
            await asyncio.sleep(1)

        # 최종 실패 처리
        if not offers:
            # result가 dict 에러 메시지를 포함할 수도 있으니 노출
            err_msg = None
            if isinstance(result, dict) and result.get("error"):
                err_msg = str(result.get("error"))
            new_state = dict(state)
            if err_msg:
                new_state["error"] = err_msg
            if os.getenv("FLIGHT_AGENT_USE_SAMPLE", "0") in ("1", "true", "True"):
                logger.warning("offers 비어있어 샘플 사용")
                new_state["offers"] = _sample_offers()
            else:
                new_state["offers"] = []
            print_phase("SHOPPING", "ERROR", new_state.get("error") or "no offers")
            return new_state

        new_state = dict(state)
        new_state["offers"] = offers
        print_phase("SHOPPING", "DONE", f"offers={len(offers)}")
        return new_state

    return shopper_node


def make_presenter_node():
    """Presenter: offers → Top-10 표 문자열 생성.

    컬럼: 순위 | 편명 | 출발지 | 출발일시 | 도착지 | 도착일시 | 등급 | 운임
    복수 segment 있는 경우 첫 segment 기준 + (추가 segment 수) 표기.
    """
    async def presenter_node(state: FlightSearchState) -> FlightSearchState:
        print_phase("PRESENTING", "START", "결과 표 생성")
        offers_list = state.get("offers", [])
        if not isinstance(offers_list, list):
            offers_len = 0
        else:
            offers_len = len(offers_list)
        logger.debug("presenter_node 시작: has_error=%s offers_len=%d", bool(state.get("error")), offers_len)
        if state.get("error") and not state.get("offers"):
            msg = [
                "오류로 인해 항공권을 불러오지 못했습니다.",
                f"서버 메시지: {state['error']}",
                "(FLIGHT_AGENT_USE_SAMPLE=1 설정 시 샘플 데이터 사용)"
            ]
            new_state = dict(state)
            new_state["result_table"] = []
            new_state["result_table_pretty"] = "\n".join(msg)
            logger.debug("presenter_node 종료(에러 경로): table_lines=%d", len(msg))
            print_phase("PRESENTING", "ERROR", "error state")
            return new_state

        offers = state.get("offers", [])
        if not isinstance(offers, list):
            offers = []

        def total_price(o: Dict[str, Any]) -> float:
            p = o.get("price") if isinstance(o, dict) else None
            if isinstance(p, dict):
                for k in ("grandTotal", "total", "amount"):
                    v = p.get(k)
                    if v is not None:
                        try:
                            return float(v)
                        except Exception:
                            pass
            return 1e20

        top = sorted(offers, key=total_price)[:10]

        header = ["순위", "편명", "출발지", "출발일시", "도착지", "도착일시", "등급", "운임"]
        lines = [" | ".join(header), " | ".join(["----"] * len(header))]

        def first_segment(o: Dict[str, Any]):
            its = o.get("itineraries") if isinstance(o, dict) else None
            if isinstance(its, list) and its:
                it0 = its[0]
                if isinstance(it0, dict):
                    segs = it0.get("segments")
                    if isinstance(segs, list) and segs:
                        return segs[0]
            return None

        def last_segment(o: Dict[str, Any]):
            its = o.get("itineraries") if isinstance(o, dict) else None
            if isinstance(its, list) and its:
                it_last = its[-1]
                if isinstance(it_last, dict):
                    segs = it_last.get("segments")
                    if isinstance(segs, list) and segs:
                        return segs[-1]
            return None

        for rank, offer in enumerate(top, 1):
            if not isinstance(offer, dict):
                continue
            fs = first_segment(offer)
            ls = last_segment(offer)
            flight_no = "-"
            extra_seg = 0
            if fs and isinstance(fs, dict):
                cc = fs.get("carrierCode") or ""
                num = fs.get("number") or ""
                if cc or num:
                    flight_no = f"{cc}{num}"
                # 추가 segment 계산(첫 itinerary 내)
                its = offer.get("itineraries")
                if isinstance(its, list):
                    for it in its:
                        if isinstance(it, dict):
                            segs = it.get("segments")
                            if isinstance(segs, list) and len(segs) > 1:
                                extra_seg += len(segs) - 1
                if extra_seg:
                    flight_no += f"(+{extra_seg})"

            dep_air = fs.get("departure", {}).get("iataCode") if isinstance(fs, dict) else "-"
            dep_time = fs.get("departure", {}).get("at") if isinstance(fs, dict) else "-"
            arr_air = ls.get("arrival", {}).get("iataCode") if isinstance(ls, dict) else "-"
            arr_time = ls.get("arrival", {}).get("at") if isinstance(ls, dict) else "-"

            cabin = "-"
            traveler_pricings = offer.get("travelerPricings")
            if isinstance(traveler_pricings, list) and traveler_pricings:
                tp0 = traveler_pricings[0]
                if isinstance(tp0, dict):
                    fdbs = tp0.get("fareDetailsBySegment")
                    if isinstance(fdbs, list) and fdbs:
                        fd0 = fdbs[0]
                        if isinstance(fd0, dict):
                            cabin = fd0.get("cabin") or fd0.get("brandedFareLabel") or fd0.get("brandedFare") or "-"

            fare = "-"
            price = offer.get("price")
            if isinstance(price, dict):
                fare = price.get("grandTotal") or price.get("total") or price.get("amount") or "-"

            line = [str(rank), flight_no, dep_air, dep_time, arr_air, arr_time, cabin, fare]
            lines.append(" | ".join(line))
        # 함수 종료 직전 결과 저장
        new_state = dict(state)
        new_state["result_table"] = top
        new_state["result_table_pretty"] = "\n".join(lines)
        logger.debug("presenter_node 종료: top_count=%d line_count=%d", len(top), len(lines))
        print_phase("PRESENTING", "DONE", f"rows={len(top)}")
        return new_state

    return presenter_node

# ---------------------------------------------------------------------------
# 실행 함수
# ---------------------------------------------------------------------------
async def run_agent(user_input: Dict[str, Any]):
    tools = await get_tools()

    use_graph = os.getenv("FLIGHT_AGENT_USE_GRAPH", "0") in ("1", "true", "True")
    planner_fn = make_planner_node()
    shopper_fn = make_shopper_node(tools)
    presenter_fn = make_presenter_node()

    if use_graph:
        logger.debug("FLIGHT_AGENT_USE_GRAPH=1 → LangGraph 경로 사용")
        graph = StateGraph(dict)
        graph.add_node("planner", planner_fn)
        graph.add_node("shopper", shopper_fn)
        graph.add_node("presenter", presenter_fn)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "shopper")
        graph.add_edge("shopper", "presenter")
        try:
            graph.set_entry_point("planner")  # type: ignore[attr-defined]
            graph.set_finish_point("presenter")  # type: ignore[attr-defined]
        except AttributeError:
            pass
        workflow = graph.compile()
        initial_state: FlightSearchState = {"user_input": user_input}
        try:
            result = await workflow.ainvoke(initial_state)
            if isinstance(result, list):
                last_dict = None
                for item in reversed(result):
                    if isinstance(item, dict):
                        last_dict = item
                        break
                result = last_dict or {}
            if not isinstance(result, dict):
                logger.error("workflow 결과 타입 이상: %s", type(result))
                result = {}
        except Exception as e:
            logger.exception("LangGraph 실행 오류, 수동 파이프라인 fallback: %s", e)
            result = {}
            use_graph = False  # fallback 아래 수동 경로로 실행
        state = result  # type: ignore[assignment]

    if not use_graph:
        # --- 수동 직렬 파이프라인 (신뢰성 우선) ---
        state: FlightSearchState = {"user_input": user_input}
        try:
            state = planner_fn(state)
            state = await shopper_fn(state)
            state = await presenter_fn(state)
        except Exception as e:
            print_phase("PIPELINE", "ERROR", str(e))
            logger.exception("수동 파이프라인 실행 중 오류: %s", e)

    # 출력 단계
    print("\n=== 최종 Top-10 항공권 ===")
    table = state.get("result_table_pretty")
    if table:
        print(table)
    else:
        offers = state.get("offers", [])
        if isinstance(offers, list) and offers:
            header = ["순위", "편명", "운임"]
            lines = [" | ".join(header), " | ".join(["----"] * len(header))]
            def _price(o):
                if isinstance(o, dict):
                    p = o.get("price")
                    if isinstance(p, dict):
                        for k in ("grandTotal", "total", "amount"):
                            v = p.get(k)
                            if v is not None:
                                try:
                                    return float(v)
                                except Exception:
                                    pass
                return 1e20
            for i, o in enumerate(sorted(offers, key=_price)[:10], 1):
                if not isinstance(o, dict):
                    continue
                seg = None
                its = o.get("itineraries") if isinstance(o, dict) else None
                if isinstance(its, list) and its:
                    it0 = its[0]
                    if isinstance(it0, dict):
                        segs = it0.get("segments")
                        if isinstance(segs, list) and segs:
                            seg = segs[0]
                flight_no = "-"
                if isinstance(seg, dict):
                    cc = seg.get("carrierCode") or ""
                    num = seg.get("number") or ""
                    if cc or num:
                        flight_no = f"{cc}{num}"
                fare = "-"
                p = o.get("price") if isinstance(o, dict) else None
                if isinstance(p, dict):
                    fare = p.get("grandTotal") or p.get("total") or p.get("amount") or "-"
                lines.append(" | ".join([str(i), flight_no, fare]))
            print("\n".join(lines))
        else:
            print("(표시할 결과가 없습니다 - presenter 및 fallback 모두 실패)")
    # 디버깅을 위해 에러 메시지 별도 표시
    if state.get("error"):
        print(f"[오류] {state['error']}")

# ---------------------------------------------------------------------------
# CLI 인터랙션
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("항공권 검색 파라미터를 입력하세요. (Enter 시 기본값)")
    origin = input("출발지(공항코드, 예: ICN): ") or "ICN"
    destination = input("도착지(공항코드, 예: JFK): ") or "JFK"
    departureDate = input("출발일(YYYY-MM-DD, 예: 2025-10-01): ") or "2025-10-01"
    adults = input("성인 인원수(기본 1): ") or "1"
    try:
        adults = int(adults)  # type: ignore[assignment]
    except ValueError:
        adults = 1  # type: ignore[assignment]

    user_input = {
        "origin": origin,
        "destination": destination,
        "departureDate": departureDate,
        "adults": adults,
    }
    asyncio.run(run_agent(user_input))
