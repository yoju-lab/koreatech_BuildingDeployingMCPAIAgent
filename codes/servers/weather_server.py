# servers/weather_server.py
"""
FastMCP 기반 HTTP 전용 MCP 서버.
- OpenWeatherMap Current/Forecast API 호출
- Tools:
    - weather.current(city, country=None, units="metric", lang="kr")
    - weather.forecast(city, country=None, days=3, units="metric", lang="kr")
환경:
    OPENWEATHER_API_KEY  (필수)
실행:
    python servers/weather_server.py
    → http://127.0.0.1:8000/mcp (streamable_http)
"""
from fastmcp import FastMCP
import httpx, os
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

# --- .env 로드: 스크립트 상위 폴더의 .env를 명시적으로 찾고, 기존 env를 덮어쓰기 ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

mcp = FastMCP("weather")  # MCP 서버 이름

def _require_key() -> str:
    key = os.getenv("OPENWEATHER_API_KEY")
    if not key:
        raise RuntimeError("환경변수 OPENWEATHER_API_KEY가 설정되어 있지 않습니다.")
    return key

async def _get_json(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    # HTTP 타임아웃/오류 처리
    timeout = httpx.Timeout(10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()

def _iso_from_unix_with_offset(unix_ts: int, offset_seconds: int) -> str:
    tz = timezone(timedelta(seconds=offset_seconds))
    return datetime.fromtimestamp(unix_ts, tz).isoformat()

@mcp.tool
async def current(
    city: str,
    country: Optional[str] = None,
    units: str = "metric",   # metric(℃, m/s) | imperial(℉, mph)
    lang: str = "kr"         # OWM 언어 코드(한국어: "kr")
) -> Dict[str, Any]:
    """
    OpenWeatherMap 'Current Weather' 호출.
    city="Seoul", country="KR" 형태 권장(국가코드 생략 가능).
    """
    key = _require_key()
    q = f"{city},{country}" if country else city
    params = {"q": q, "appid": key, "units": units, "lang": lang}
    url = "https://api.openweathermap.org/data/2.5/weather"

    try:
        data = await _get_json(url, params)
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}

    weather = (data.get("weather") or [{}])[0]
    main = data.get("main", {})
    wind = data.get("wind", {})
    sys = data.get("sys", {})
    coord = data.get("coord", {})
    tz_offset = data.get("timezone", 0)
    dt_unix = data.get("dt")
    iso_time = _iso_from_unix_with_offset(dt_unix, tz_offset) if dt_unix else None

    return {
        "city": data.get("name"),
        "country": sys.get("country"),
        "coord": {"lat": coord.get("lat"), "lon": coord.get("lon")},
        "temperature": main.get("temp"),
        "feels_like": main.get("feels_like"),
        "humidity": main.get("humidity"),
        "pressure": main.get("pressure"),
        "wind_speed": wind.get("speed"),   # metric: m/s, imperial: mph
        "wind_deg": wind.get("deg"),
        "weather": weather.get("description"),
        "icon": weather.get("icon"),
        "time_local": iso_time,
        "units": units,
        "source": "openweathermap",
    }

@mcp.tool
async def forecast(
    city: str,
    country: Optional[str] = None,
    days: int = 3,           # 1~5 권장(OWM 5일/3시간 예보)
    units: str = "metric",
    lang: str = "kr"
) -> Dict[str, Any]:
    """
    OpenWeatherMap '5 day / 3 hour' 예보를 일자별 요약으로 반환.
    - 각 날짜별 평균/최저/최고 기온과 대표 날씨(최빈값) 제공
    """
    key = _require_key()
    days = 1 if days < 1 else 5 if days > 5 else days

    q = f"{city},{country}" if country else city
    params = {"q": q, "appid": key, "units": units, "lang": lang}
    url = "https://api.openweathermap.org/data/2.5/forecast"

    try:
        data = await _get_json(url, params)
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"error": str(e)}

    lst: List[Dict[str, Any]] = data.get("list", [])
    city_info = data.get("city", {})
    tz_offset = city_info.get("timezone", 0)
    name = city_info.get("name") or city
    country_code = city_info.get("country")
    coord = city_info.get("coord", {})

    # 로컬 날짜 기준으로 그룹핑
    by_date: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in lst:
        dt_unix = item.get("dt")
        if dt_unix is None:
            continue
        iso = _iso_from_unix_with_offset(dt_unix, tz_offset)
        local_date = iso[:10]  # YYYY-MM-DD
        by_date[local_date].append(item)

    summaries = []
    for d in sorted(by_date.keys())[:days]:
        items = by_date[d]
        temps = [it.get("main", {}).get("temp") for it in items if it.get("main")]
        temp_min = min([it.get("main", {}).get("temp_min") for it in items if it.get("main")], default=None)
        temp_max = max([it.get("main", {}).get("temp_max") for it in items if it.get("main")], default=None)
        descs = [ (it.get("weather") or [{}])[0].get("description") for it in items ]
        cnt = Counter([x for x in descs if x])
        desc = cnt.most_common(1)[0][0] if cnt else None

        summaries.append({
            "date": d,
            "temp_avg": (sum(t for t in temps if t is not None) / len(temps)) if temps else None,
            "temp_min": temp_min,
            "temp_max": temp_max,
            "weather": desc,
        })

    return {
        "city": name,
        "country": country_code,
        "coord": {"lat": coord.get("lat"), "lon": coord.get("lon")},
        "units": units,
        "days": summaries,
        "source": "openweathermap",
    }

if __name__ == "__main__":
    # HTTP 전용(Streamable HTTP)으로 기동합니다.
    # 엔드포인트: http://127.0.0.1:8000/mcp
    mcp.run(transport="http", host="127.0.0.1", port=8000, path="/mcp")
