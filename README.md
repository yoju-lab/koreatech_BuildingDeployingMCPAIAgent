# koreatech_langgraph_multiagent
langgraph for developing AI multi-agent services

## codes 디렉토리 구성 요약
- `langgraph_01_개발_환경_준비.md`, `langgraph_02_개념_설명.md`: LangGraph 개발 환경 준비 및 개념 설명 문서
- `수업 노트.docx`, `수업 노트_01.docx`, `수업 노트_02.docx`: 수업 관련 워드 문서
- `n_projects/PROJECT_정의서.md`: 프로젝트 정의서
- `n10_simple_langgraph_chat/`: LangGraph 기반의 간단한 챗봇 예제 (Jupyter 노트북, 설명 문서 포함)
- `n20_langgraph_chat_with_tools/`: 도구와 연동된 LangGraph 챗봇 예제 (여러 Jupyter 노트북, 설명 문서 포함)
- `n30_langgraph_chat_with_memory/`: 메모리 기능이 포함된 LangGraph 챗봇 예제 (Jupyter 노트북, 설명 문서 포함)
- `n40_langgraph_chat_with_human_in_the_loop/`: Human-in-the-loop 구조의 LangGraph 챗봇 예제 (Jupyter 노트북, 설명 문서 포함)
- `n50_langgraph_chat_with_custom_state/`: 커스텀 상태 관리가 포함된 LangGraph 챗봇 예제 (Jupyter 노트북, 설명 문서 포함)
- `n60_agents/`: 다양한 에이전트(Jupyter 노트북, 설명 문서 포함)
  - 예: intent, food, activity, keyword, place, summary, time, season, weather, intent_unsupported agent 등
- `n60_agents_ipynb/`: 다양한 에이전트의 Jupyter 노트북 모음 (코드 및 설명서)
- `n70_agents_projects/`: 에이전트 프로젝트 예제 및 실행 코드, 테스트, 설정 파일 등 포함
- `n80_agents_supervisor/`: 에이전트 supervisor 관련 Jupyter 노트북

## 실행 방법
### 1. Docker 환경에서 실행
```bash
cd dockers
sudo docker compose up --build
```
- LangGraph Studio: http://localhost:8000

### 2. 로컬 개발 환경에서 실행 (Jupyter Lab)
```bash
pip install -r requirements.txt
jupyter lab
```

### 3. Streamlit 앱 실행 예시 (예: streamlit_app.py가 있을 경우)
```bash
streamlit run streamlit_app.py
```
