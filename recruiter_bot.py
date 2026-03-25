import asyncio
import random
import datetime
import json
import os
import pickle
import re

from playwright.async_api import async_playwright
from openai import OpenAI

import gspread
from oauth2client.service_account import ServiceAccountCredentials

# httpx가 시스템 로케일 한글을 헤더에 ASCII 인코딩하려다 실패하는 문제 패치
import httpx._models as _httpx_models
_orig_normalize = _httpx_models._normalize_header_value
def _patched_normalize(value, encoding):
    try:
        return _orig_normalize(value, encoding)
    except UnicodeEncodeError:
        if isinstance(value, str):
            return value.encode("ascii", errors="ignore")
        raise
_httpx_models._normalize_header_value = _patched_normalize

client = OpenAI(timeout=30.0)


# =========================
# 임베딩 저장소
# =========================

EMBEDDING_STORE_PATH = "/Users/cng/recruiter_embeddings.pkl"


def load_embedding_store():
    if os.path.exists(EMBEDDING_STORE_PATH):
        with open(EMBEDDING_STORE_PATH, "rb") as f:
            return pickle.load(f)
    return []


def save_embedding_store(store):
    with open(EMBEDDING_STORE_PATH, "wb") as f:
        pickle.dump(store, f)


def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text[:8000]
    )
    return response.data[0].embedding



def extract_fingerprint(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""다음 이력서에서 아래 정보를 추출하세요.

이력서:
{text[:8000]}

반드시 아래 JSON 형식으로만 응답하라.
{{
  "companies": ["회사명1", "회사명2", "회사명3"],
  "school": "최종학력 학교명 또는 null",
  "major": "전공 또는 null"
}}

주의:
- companies: 이력서에 등장하는 순서 그대로 회사명 배열 (날짜 기준으로 재정렬하지 말 것)
- 인턴·프리랜서·강연 등 모든 경력 포함
- 회사명은 공식 명칭으로 (주식회사 등 법인격 제외)
- school은 최종학력 학교명만"""
            }],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[extract_fingerprint 오류] {e}")
        return {"companies": [], "school": None, "major": None}


def fingerprint_match(fp1, fp2):
    """두 fingerprint가 동일인인지 판단."""
    companies1 = fp1.get("companies", [])
    companies2 = fp2.get("companies", [])

    if not companies1 or not companies2:
        return False

    # 회사 목록 순서까지 동일 → 동일인
    if companies1 == companies2:
        return True

    set1 = set(companies1)
    set2 = set(companies2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    jaccard = intersection / union if union > 0 else 0

    # 80% 이상 겹치면 학교 무관하게 동일인
    if jaccard >= 0.8:
        return True

    # 60% 이상 겹치고 학교도 일치하면 동일인
    if jaccard >= 0.6:
        school1 = fp1.get("school")
        school2 = fp2.get("school")
        if school1 and school2 and school1 == school2:
            return True

    return False


def find_duplicate(new_fingerprint, store):
    if not new_fingerprint or not new_fingerprint.get("companies"):
        return None
    for entry in store:
        fp = entry.get("fingerprint")
        if fp and fingerprint_match(new_fingerprint, fp):
            return entry
    return None


# =========================
# Google Sheet 연결
# =========================

def connect_sheets():

    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]

    creds = ServiceAccountCredentials.from_json_keyfile_name(
        "google_credentials.json",
        scope
    )

    client_gsheet = gspread.authorize(creds)

    spreadsheet = client_gsheet.open("AI Recruiter")

    position_sheets = {
        "Recruiting Manager": spreadsheet.worksheet("Recruiting Manager"),
        "Product Engineer (AI Contents)": spreadsheet.worksheet("Product Engineer (AI Contents)"),
        "사업 PM (정부사업)": spreadsheet.worksheet("사업 PM (정부사업)"),
        "프로덕트 매니저 (PM)": spreadsheet.worksheet("프로덕트 매니저 (PM)"),
        "Sales Manager": spreadsheet.worksheet("Sales Manager"),
    }

    recommend_sheet = spreadsheet.worksheet("Recommended")

    return position_sheets, recommend_sheet


# =========================
# 기존 후보 로드
# =========================

def load_existing_ids(sheet):

    existing_ids = set()

    rows = sheet.get_all_values()

    for r in rows[1:]:

        if len(r) >= 7:

            url = r[6]

            if "preview_user_hash=" in url:
                # Wanted
                cid = url.split("preview_user_hash=")[-1]
                existing_ids.add(cid)

            elif "rememberapp.co.kr/profiles/" in url:
                # Remember
                match = re.search(r'/profiles/(\d+)', url)
                if match:
                    existing_ids.add(match.group(1))

    return existing_ids


# =========================
# GPT 평가 - Recruiting Manager
# =========================

def evaluate_recruiting_manager(text):

    prompt = f"""
당신은 스타트업 리크루팅 전문가입니다.

후보자의 이력서를 분석하여 추천 여부를 판단하세요.

【사전 필터 - 아래 조건 중 하나라도 해당하면 즉시 비추천(false)】

1. 대학교 입학년도가 2011년 이전인 경우
   - 입학년도 추정 방법: 졸업년도에서 4년을 뺀다. 졸업년도가 없으면 최초 경력 시작년도에서 역산한다.
   - 입학년도를 알 수 없는 경우에는 필터를 적용하지 않는다.
2. C&B, 급여(Payroll), 보상체계 설계·관리, 총무(GA)가 커리어의 주요 축을 이루는 경우
   - 채용이 부수적인 업무로만 포함된 제너럴리스트 HR은 비추천
   - 단, C&B/총무 경험이 있더라도 채용 전문가로 전환한 명확한 커리어 변화가 있으면 예외
   - 주의: 채용 담당자가 성과평가 지원·보상 재조정 서포트를 부수적으로 수행한 경우는 이 필터를 적용하지 않는다
   - 채용(Recruiting/TA) 직무명을 유지하면서 일부 HR 업무를 겸한 경우도 필터 적용 제외

강한 감점 신호

- HR 전반(평가, 보상, 노무, 총무 등)을 수행하는 제너럴리스트 커리어가 중심인 경우
- 채용이 HR 업무 중 일부로만 수행된 경우
- 이직 이후에도 지속적으로 HR 전반 역할을 수행한 경우
- C&B / Payroll / 급여체계 / 보상설계 / 총무가 주요 성과로 서술된 경우
- 단, 직무명이 Recruiter·TA·채용담당이고 채용 성과가 주를 이루는 경우 위 신호는 무시한다


추천 판단 기준

사전 필터를 통과한 경우에만 아래 조건을 평가한다.

【필수 조건 - 반드시 충족해야 추천 가능】

1. 인하우스 채용 담당자로서 다이렉트 소싱을 직접 수행한 경험이 있어야 한다.
   - 인하우스: 일반 기업(IT·스타트업·제조·서비스 등) 내부 HR팀/채용팀 소속
   - 에이전시(서치펌·헤드헌팅·채용 대행사) 경력만 있는 경우 미충족
     판별 기준: "클라이언트사", "의뢰기업", "후보자 소개" 표현 / 회사 자체가 채용 대행 업종
   - 에이전시 경력이 있어도 인하우스 경력에서 소싱 경험 있으면 인정

2. 다이렉트 소싱 활동이 이력서에 언급되어 있어야 한다.
   아래 중 하나 이상이 이력서에 언급되면 인정:
   - 소싱으로 컨택·커피챗·메시지 발송한 인원 수 (예: "100명 컨택", "50명 커피챗")
   - 소싱으로 채용까지 이어진 건수 (예: "다이렉트 소싱으로 10명 입사", "두자릿수 신규입사자 확보")
   - 소싱 채널·전략 실행 내용 (예: "LinkedIn·리멤버 활용 아웃바운드 운영", "Direct Sourcing 기반 후보자 발굴")
   - 소싱 퍼널 수치 (컨택→수락→합격 흐름)

필수 조건 충족 후, 다음 조건을 2가지 이상 만족하는 경우에만 추천한다.

1. IT / 스타트업 / SaaS 환경 채용 경험
2. 채용 퍼널 개선 또는 채용 데이터 기반 개선 경험
3. 채용 전문가로서 커리어를 발전시키려는 명확한 경험 또는 방향성


signals 후보

OUTBOUND_SOURCING
IT_RECRUITING
STARTUP_ENV
FUNNEL_OPTIMIZATION
RECRUITING_SPECIALIST


후보 이력서

{text}


반드시 아래 JSON 형식으로만 응답하라.

{{
 "recommend": true 또는 false,
 "signals": ["SIGNAL1","SIGNAL2"],
 "note": "15단어 이하의 짧은 판단 이유"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    result = json.loads(response.choices[0].message.content)

    return result["recommend"], result["signals"], result["note"]


# =========================
# GPT 평가 - Product Engineer (AI Contents)
# =========================

def evaluate_product_engineer(text):

    prompt = f"""
당신은 스타트업 채용 전문가입니다.

후보자의 이력서를 분석하여 AI 콘텐츠 제작 관련 경험을 평가하세요.

【사전 필터 - 아래 조건 중 하나라도 해당하면 즉시 비추천(false)】

1. 아래 AI 생성형 도구를 실제 업무 또는 프로젝트에 활용한 구체적 경험이 없는 경우
   - Midjourney, Stable Diffusion, ControlNet, LoRA, ComfyUI, Generative AI workflow, AI pipeline, Prompt engineering
   - 단순히 "AI 도구 관심 있음", "학습 중" 등의 서술은 미충족
2. 일반 디자이너로서 AI 도구 활용 없이 Photoshop·Illustrator·Figma 등만 사용한 경우

【추천 기준】

사전 필터를 통과한 후, 아래 강한 긍정 신호 중 2개 이상 충족하는 경우에만 추천한다.

- Midjourney 실무 활용 경험
- Stable Diffusion 실무 활용 경험
- ControlNet 실무 활용 경험
- LoRA 실무 활용 경험
- ComfyUI 실무 활용 경험
- Generative AI workflow 설계·운영 경험
- AI pipeline 구축 경험
- Prompt engineering 실무 적용 경험


signals 후보

MIDJOURNEY
STABLE_DIFFUSION
CONTROLNET
LORA
COMFYUI
GENERATIVE_AI_WORKFLOW
AI_PIPELINE
PROMPT_ENGINEERING
GENERAL_DESIGNER


후보 이력서

{text}


반드시 아래 JSON 형식으로만 응답하라.

{{
 "recommend": true 또는 false,
 "signals": ["SIGNAL1","SIGNAL2"],
 "note": "15단어 이하의 짧은 판단 이유"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    result = json.loads(response.choices[0].message.content)

    return result["recommend"], result["signals"], result["note"]


# =========================
# GPT 평가 - 사업 PM (정부사업)
# =========================

def evaluate_business_pm(text):

    prompt = f"""
당신은 스타트업 채용 전문가입니다.

후보자의 이력서를 분석하여 정부지원사업 PM 적합 여부를 판단하세요.

【사전 필터 - 아래 조건 중 하나라도 해당하면 즉시 비추천(false)】

1. 정부지원사업 경험이 전혀 없는 경우
2. 정부지원사업 1 Cycle 이상 전 주기(기획~정산)를 직접 담당한 경험이 없는 경우
   - 단순 서포트·행정 보조·일부 참여는 미충족
3. 이력서에 본인의 역할이 구체적으로 서술되지 않은 경우
   - "~사업을 수행했다", "~과제에 참여했다"로만 끝나는 경우 즉시 비추천
   - 본인이 어떤 역할을 맡았는지 명시된 경우만 인정

【추천 기준】

사전 필터를 통과한 후, 아래 3가지 조건 중 2가지 이상 충족하는 경우에만 추천한다.

1. R&D 정부지원사업 / 연구개발과제 / 국책과제 관리 경험 (2년 이상)
2. 과제 선정 성공 경험 (제안서 작성 및 수주)
3. 정부지원과제 관리 시스템 활용 경험 (이지바로, RCMS, Smtech 등)

강한 감점 신호

- 정부사업 경험이 간접적이거나 일부에 불과한 경우
- 역할이 행정·서류 처리 위주인 경우


signals 후보

GOV_PROJECT_MANAGEMENT
FULL_CYCLE_EXPERIENCE
BUDGET_MANAGEMENT
PROPOSAL_WRITING
GOV_SYSTEM_EXPERIENCE
PUBLIC_NETWORK
STARTUP_EXPERIENCE


후보 이력서

{text}


반드시 아래 JSON 형식으로만 응답하라.

{{
 "recommend": true 또는 false,
 "signals": ["SIGNAL1","SIGNAL2"],
 "note": "15단어 이하의 짧은 판단 이유"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    result = json.loads(response.choices[0].message.content)

    return result["recommend"], result["signals"], result["note"]


# =========================
# GPT 평가 - 프로덕트 매니저 (PM)
# =========================

def evaluate_product_manager(text):

    prompt = f"""
당신은 스타트업 채용 전문가입니다.

후보자의 이력서를 분석하여 프로덕트 매니저 적합 여부를 판단하세요.

【사전 필터 - 아래 조건 중 하나라도 해당하면 즉시 비추천(false)】

1. PM 직무 경험 없이 서비스 기획·운영·콘텐츠 기획만 있는 경우
   - 직책이 기획자·운영자·MD 중심이고 제품 로드맵·스펙 정의 경험이 없는 경우
2. 데이터 분석 기반 의사결정 경험이 전혀 없는 경우
   - 지표 없이 직관·감각 중심 의사결정만 서술된 경우

【추천 기준】

사전 필터를 통과한 후, 아래 조건 중 2가지 이상 충족하는 경우에만 추천한다.

1. 짧은 실험 사이클 경험 (A/B 테스트, 스프린트 등)
2. 개발 / 디자인 / 세일즈 크로스펑셔널 협업 경험
3. 지표 개선 성과를 수치로 제시한 경험
4. 스타트업 PM 경험

강한 긍정 신호

- 데이터 분석 및 실험 기반 제품 개선
- AI 도구 활용한 리서치 / 분석 경험
- 실패와 레슨런을 명확히 서술한 경험

강한 감점 신호

- 기획 중심, 실행/실험 경험 없음
- 대기업 기획/운영 중심 경력


signals 후보

DATA_DRIVEN
AB_TESTING
CROSS_FUNCTIONAL
METRIC_IMPROVEMENT
AI_TOOL_USAGE
STARTUP_PM
HYPOTHESIS_VALIDATION


후보 이력서

{text}


반드시 아래 JSON 형식으로만 응답하라.

{{
 "recommend": true 또는 false,
 "signals": ["SIGNAL1","SIGNAL2"],
 "note": "15단어 이하의 짧은 판단 이유"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    result = json.loads(response.choices[0].message.content)

    return result["recommend"], result["signals"], result["note"]


# =========================
# GPT 평가 - Sales Manager
# =========================

def evaluate_sales_manager(text):

    prompt = f"""
당신은 스타트업 채용 전문가입니다.

후보자의 이력서를 분석하여 Sales Manager 적합 여부를 판단하세요.

【사전 필터 - 아래 조건 중 하나라도 해당하면 즉시 비추천(false)】

1. B2B 또는 B2G 영업 경력이 2년 미만인 경우
   - B2C 리테일·소비재 영업만 있는 경우도 미충족
2. 데이터 기반 영업 성과 수치가 이력서에 전혀 명시되지 않은 경우
   - 단순 "목표 달성", "매출 기여" 등의 서술만 있는 경우 미충족
   - 매출액, 성장률, 고객 수, 계약 건수, 절감 비용 등 구체적 수치가 기재된 경우만 충족

【추천 기준】

사전 필터를 통과한 후, 아래 5가지 조건 중 3가지 이상 충족하는 경우에만 추천한다.

1. 오프라인 사업자 대상 SaaS / 소프트웨어 / 하드웨어 솔루션 영업 경험
2. B2G / 공공기관 / 지자체 예산 반영 및 매출 성과 경험
3. 파트너(영업대행) 관리를 통한 성과 경험
4. CRM 도구 활용 경험 (Salesforce, HubSpot, 자체 CRM 등)
5. 스타트업 근무 경험


signals 후보

B2B_SALES
B2G_SALES
SOLUTION_SALES
PARTNER_MANAGEMENT
CRM_EXPERIENCE
STARTUP_EXPERIENCE
DATA_DRIVEN_SALES


후보 이력서

{text}


반드시 아래 JSON 형식으로만 응답하라.

{{
 "recommend": true 또는 false,
 "signals": ["SIGNAL1","SIGNAL2"],
 "note": "15단어 이하의 짧은 판단 이유"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    result = json.loads(response.choices[0].message.content)

    return result["recommend"], result["signals"], result["note"]


# =========================
# 모달 스크롤
# =========================

async def scroll_modal(page):

    await page.evaluate("""
        async () => {
            const modal = document.querySelector("div[role='dialog']");
            if (!modal) return;

            let scrollEl = modal;
            const children = modal.querySelectorAll('*');
            for (const el of children) {
                const style = window.getComputedStyle(el);
                if (style.overflowY === 'auto' || style.overflowY === 'scroll') {
                    if (el.scrollHeight > el.clientHeight) {
                        scrollEl = el;
                        break;
                    }
                }
            }

            let lastTop = -1;
            while (true) {
                scrollEl.scrollTop += 600;
                await new Promise(r => setTimeout(r, 300));
                if (scrollEl.scrollTop === lastTop) break;
                lastTop = scrollEl.scrollTop;
            }
        }
    """)

    await asyncio.sleep(1)


# =========================
# 포지션별 후보 처리
# =========================

async def process_position(page, position_sheet, recommend_sheet, existing_ids, today, position_name, base_url, evaluate_fn, embedding_store):

    for page_num in range(1, 4):

        print(f"\n[{position_name}] 페이지 이동: {page_num}")

        listing_url = base_url.format(page_num)

        await page.goto(listing_url)
        await asyncio.sleep(random.uniform(3, 6))

        cards = page.locator("div[role='button']")
        total_cards = await cards.count()

        print(f"카드 수: {total_cards}")

        if total_cards == 0:
            print("카드 없음, 다음 페이지로")
            continue

        for i in range(total_cards):

            try:

                # 모달이 열려있으면 닫기
                if "preview_user_hash=" in page.url:
                    await page.keyboard.press("Escape")
                    await asyncio.sleep(1)

                # 카드 재탐색 후 클릭
                cards = page.locator("div[role='button']")
                current_count = await cards.count()

                if i >= current_count:
                    print(f"카드 인덱스 초과 (i={i}, count={current_count}), 다음 페이지로")
                    break

                card = cards.nth(i)
                await card.scroll_into_view_if_needed()
                await card.click()
                await asyncio.sleep(random.uniform(1, 3))

                candidate_url = page.url

                if "preview_user_hash=" not in candidate_url:
                    try:
                        await page.keyboard.press("Escape")
                    except Exception:
                        pass
                    continue

                candidate_id = candidate_url.split("preview_user_hash=")[-1]

                if candidate_id in existing_ids:
                    print(f"이미 검토한 후보 스킵: {candidate_id[:12]}...")
                    await page.keyboard.press("Escape")
                    await asyncio.sleep(1)
                    continue

                modal = page.locator("div[role='dialog']")

                try:
                    await modal.wait_for(timeout=6000)
                except Exception:
                    print("모달 대기 실패, 스킵")
                    await page.keyboard.press("Escape")
                    continue

                # 모달 전체 스크롤
                await scroll_modal(page)

                text = await modal.inner_text(timeout=10000)

                if not text.strip():
                    print("텍스트 없음, 스킵")
                    await page.keyboard.press("Escape")
                    continue

                # fingerprint 생성 및 중복 체크
                new_fingerprint = extract_fingerprint(text)
                dup = find_duplicate(new_fingerprint, embedding_store)
                if dup:
                    print(f"중복 추정 스킵: {candidate_id[:12]}... (유사 후보: {dup['candidate_id'][:12]}... / {dup['platform']})")
                    existing_ids.add(candidate_id)
                    await page.keyboard.press("Escape")
                    await asyncio.sleep(1)
                    continue

                recommend, signals, note = evaluate_fn(text)

                signals_str = ",".join(signals)

                # fingerprint 저장
                embedding_store.append({
                    "candidate_id": candidate_id,
                    "platform": "wanted",
                    "position": position_name,
                    "date": today,
                    "fingerprint": new_fingerprint
                })
                save_embedding_store(embedding_store)

                # 최근 재직회사 추출
                companies = new_fingerprint.get("companies", []) if new_fingerprint else []
                recent_company = companies[0] if companies else ""

                # 포지션 시트에 기록
                position_sheet.append_row([
                    today,
                    position_name,
                    "추천" if recommend else "비추천",
                    signals_str,
                    note,
                    recent_company,
                    candidate_url
                ])

                # Recommended 기록
                if recommend:
                    recommend_sheet.append_row([
                        today,
                        position_name,
                        signals_str,
                        note,
                        recent_company,
                        candidate_url
                    ])

                existing_ids.add(candidate_id)

                result_label = "추천" if recommend else "비추천"
                print(f"기록 완료 [{result_label}] {candidate_id[:12]}... | {note}")

                await page.keyboard.press("Escape")
                await asyncio.sleep(random.uniform(1, 4))

            except Exception as e:

                print(f"후보 처리 중 에러: {e}")

                try:
                    await page.keyboard.press("Escape")
                except Exception:
                    pass

                await asyncio.sleep(2)


# =========================
# 포지션별 후보 처리 - 리멤버
# =========================

async def process_position_remember(page, position_sheet, recommend_sheet, existing_ids, today, position_name, base_url, evaluate_fn, embedding_store):

    for page_num in range(1, 6):

        print(f"\n[{position_name} / 리멤버] 페이지 이동: {page_num}")

        listing_url = base_url.format(page_num)

        try:
            await page.goto(listing_url, timeout=20000)
        except Exception:
            print(f"목록 페이지 로드 타임아웃, 스킵: 페이지 {page_num}")
            continue
        await asyncio.sleep(random.uniform(3, 6))

        # 프로필 링크가 렌더링될 때까지 대기
        try:
            await page.wait_for_selector("a[href*='/profiles/']", timeout=10000)
        except Exception:
            pass

        # 프로필 링크에서 후보 ID 수집
        hrefs = await page.locator("a[href*='/profiles/']").evaluate_all(
            "els => els.map(e => e.getAttribute('href'))"
        )

        seen = set()
        candidates = []
        for href in hrefs:
            if not href:
                continue
            match = re.search(r'/profiles/(\d+)', href)
            if match:
                cid = match.group(1)
                if cid not in seen:
                    seen.add(cid)
                    candidates.append(cid)

        print(f"카드 수: {len(candidates)}")

        if not candidates:
            print("카드 없음, 종료")
            break

        for candidate_id in candidates:

            try:

                if candidate_id in existing_ids:
                    print(f"이미 검토한 후보 스킵: {candidate_id}")
                    continue

                candidate_url = f"https://career.rememberapp.co.kr/profiles/{candidate_id}?refer=search"

                try:
                    await page.goto(candidate_url, wait_until="networkidle", timeout=20000)
                except Exception:
                    print(f"페이지 로드 타임아웃, 스킵: {candidate_id}")
                    continue
                await asyncio.sleep(1)

                # 페이지 스크롤 (lazy-load 콘텐츠 로드 유도)
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(1.5)
                await page.evaluate("window.scrollTo(0, 0)")
                await asyncio.sleep(0.5)

                # 페이지 텍스트 추출
                text = await page.evaluate("document.body.textContent") or ""
                text = " ".join(text.split())

                if not text.strip():
                    print(f"텍스트 없음, 스킵: {candidate_id}")
                    continue

                print(f"[텍스트 샘플] 길이={len(text)} | {text[:300].replace(chr(10), ' ')}")

                # fingerprint 생성 및 중복 체크
                new_fingerprint = extract_fingerprint(text)
                print(f"[fingerprint] companies={new_fingerprint.get('companies')}")
                dup = find_duplicate(new_fingerprint, embedding_store)
                if dup:
                    print(f"중복 추정 스킵: {candidate_id} (유사 후보: {dup['candidate_id'][:12]}... / {dup['platform']})")
                    existing_ids.add(candidate_id)
                    continue

                recommend, signals, note = evaluate_fn(text)

                signals_str = ",".join(signals)

                # fingerprint 저장
                embedding_store.append({
                    "candidate_id": candidate_id,
                    "platform": "remember",
                    "position": position_name,
                    "date": today,
                    "fingerprint": new_fingerprint
                })
                save_embedding_store(embedding_store)

                # 최근 재직회사 추출
                companies = new_fingerprint.get("companies", []) if new_fingerprint else []
                recent_company = companies[0] if companies else ""

                # 포지션 시트에 기록
                position_sheet.append_row([
                    today,
                    position_name,
                    "추천" if recommend else "비추천",
                    signals_str,
                    note,
                    recent_company,
                    candidate_url
                ])

                # Recommended 기록
                if recommend:
                    recommend_sheet.append_row([
                        today,
                        position_name,
                        signals_str,
                        note,
                        recent_company,
                        candidate_url
                    ])

                existing_ids.add(candidate_id)

                result_label = "추천" if recommend else "비추천"
                print(f"기록 완료 [{result_label}] {candidate_id} | {note}")

                await asyncio.sleep(random.uniform(1, 3))

            except Exception as e:
                import traceback
                print(f"후보 처리 중 에러: {e}")
                traceback.print_exc()
                await asyncio.sleep(2)


# =========================
# 메인 실행
# =========================

SCAN_INTERVAL_HOURS = 6

POSITIONS = [

    # ── 리멤버 ────────────────────────────────────────────────────────────────
    {
        "platform": "remember",
        "sheet_name": "Recruiting Manager",
        "base_url": "https://career.rememberapp.co.kr/profiles/search?sort=recommendation_score%3Adesc&per=20&comprehensiveSearch=%7B%7D&categorySearch=%7B%7D&filterOptions=%7B%22finalDegreeGroup%22%3A%5B%5D%2C%22careerYear%22%3A%7B%22gte%22%3A3%7D%2C%22jobCategory%22%3A%5B%7B%22level1%22%3A%22HR%C2%B7%EC%B4%9D%EB%AC%B4%22%2C%22level2%22%3A%22%EC%B1%84%EC%9A%A9%22%7D%5D%7D&recommendationSearch=%7B%22recommender%22%3A%22job_posting%22%2C%22value%22%3A%7B%22jobPostingId%22%3A293636%7D%7D&folderId=841931&page={}",
        "evaluate_fn": evaluate_recruiting_manager
    },
    # 사업 PM 일시중단
    # {
    #     "platform": "remember",
    #     "sheet_name": "사업 PM (정부사업)",
    #     "base_url": "https://career.rememberapp.co.kr/profiles/search?...",
    #     "evaluate_fn": evaluate_business_pm
    # },
    {
        "platform": "remember",
        "sheet_name": "프로덕트 매니저 (PM)",
        "base_url": "https://career.rememberapp.co.kr/profiles/search?sort=recommendation_score%3Adesc&per=20&recommendationSearch=%7B%22recommender%22%3A%22job_posting%22%2C%22value%22%3A%7B%22jobPostingId%22%3A295257%7D%7D&filterOptions=%7B%22finalDegreeGroup%22%3A%5B%5D%2C%22careerYear%22%3A%7B%22gte%22%3A5%2C%22lte%22%3A10%7D%2C%22jobCategory%22%3A%5B%7B%22level1%22%3A%22%EA%B2%BD%EC%98%81%C2%B7%EC%A0%84%EB%9E%B5%C2%B7%EA%B8%B0%ED%9A%8D%22%2C%22level2%22%3A%22PM%C2%B7PMO%28%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8%29%22%7D%2C%7B%22level1%22%3A%22%EC%84%9C%EB%B9%84%EC%8A%A4%EA%B8%B0%ED%9A%8D%C2%B7%EC%9A%B4%EC%98%81%22%2C%22level2%22%3A%22%EC%84%9C%EB%B9%84%EC%8A%A4%EA%B8%B0%ED%9A%8D%C2%B7PM%2FPO%28%ED%94%84%EB%A1%9C%EB%8D%95%ED%8A%B8%29%22%7D%2C%7B%22level1%22%3A%22%EC%84%9C%EB%B9%84%EC%8A%A4%EA%B8%B0%ED%9A%8D%C2%B7%EC%9A%B4%EC%98%81%22%2C%22level2%22%3A%22%EC%84%9C%EB%B9%84%EC%8A%A4%EC%9A%B4%EC%98%81%22%7D%5D%7D&folderId=846136&page={}",
        "evaluate_fn": evaluate_product_manager
    },
    {
        "platform": "remember",
        "sheet_name": "프로덕트 매니저 (PM)",
        "base_url": "https://career.rememberapp.co.kr/profiles/search?sort=recommendation_score%3Adesc&per=20&recommendationSearch=%7B%22recommender%22%3A%22job_posting%22%2C%22value%22%3A%7B%22jobPostingId%22%3A291761%7D%7D&filterOptions=%7B%22finalDegreeGroup%22%3A%5B%5D%2C%22careerYear%22%3A%7B%22lte%22%3A5%7D%2C%22jobCategory%22%3A%5B%7B%22level1%22%3A%22%EA%B2%BD%EC%98%81%C2%B7%EC%A0%84%EB%9E%B5%C2%B7%EA%B8%B0%ED%9A%8D%22%2C%22level2%22%3A%22PM%C2%B7PMO%28%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8%29%22%7D%2C%7B%22level1%22%3A%22%EC%84%9C%EB%B9%84%EC%8A%A4%EA%B8%B0%ED%9A%8D%C2%B7%EC%9A%B4%EC%98%81%22%2C%22level2%22%3A%22%EC%84%9C%EB%B9%84%EC%8A%A4%EA%B8%B0%ED%9A%8D%C2%B7PM%2FPO%28%ED%94%84%EB%A1%9C%EB%8D%95%ED%8A%B8%29%22%7D%2C%7B%22level1%22%3A%22%EC%84%9C%EB%B9%84%EC%8A%A4%EA%B8%B0%ED%9A%8D%C2%B7%EC%9A%B4%EC%98%81%22%2C%22level2%22%3A%22%EC%84%9C%EB%B9%84%EC%8A%A4%EC%9A%B4%EC%98%81%22%7D%5D%7D&folderId=837170&page={}",
        "evaluate_fn": evaluate_product_manager
    },

    {
        "platform": "remember",
        "sheet_name": "Sales Manager",
        "base_url": "https://career.rememberapp.co.kr/profiles/search?sort=recommendation_score%3Adesc&page={}&per=20&comprehensiveSearch=%7B%7D&categorySearch=%7B%7D&filterOptions=%7B%22finalDegreeGroup%22%3A%5B%5D%2C%22careerYear%22%3A%7B%22gte%22%3A2%2C%22lte%22%3A10%7D%2C%22jobCategory%22%3A%5B%7B%22level1%22%3A%22%EC%98%81%EC%97%85%22%2C%22level2%22%3A%22%EC%98%81%EC%97%85+%EC%A0%84%EB%9E%B5%C2%B7%EA%B8%B0%ED%9A%8D%22%7D%2C%7B%22level1%22%3A%22%EC%98%81%EC%97%85%22%2C%22level2%22%3A%22%EA%B5%AD%EB%82%B4B2B%EC%98%81%EC%97%85%22%7D%2C%7B%22level1%22%3A%22%EC%98%81%EC%97%85%22%2C%22level2%22%3A%22%EA%B5%AD%EB%82%B4B2C%EC%98%81%EC%97%85%22%7D%2C%7B%22level1%22%3A%22%EC%98%81%EC%97%85%22%2C%22level2%22%3A%22%EA%B5%AD%EB%82%B4B2G%EC%98%81%EC%97%85%22%7D%2C%7B%22level1%22%3A%22%EC%98%81%EC%97%85%22%2C%22level2%22%3A%22IT%C2%B7%EC%86%94%EB%A3%A8%EC%85%98%EC%98%81%EC%97%85%22%7D%5D%7D&recommendationSearch=%7B%22recommender%22%3A%22job_posting%22%2C%22value%22%3A%7B%22jobPostingId%22%3A299466%7D%7D&folderId=857253",
        "evaluate_fn": evaluate_sales_manager
    },

    # ── Wanted ──────────────────────────────────────────────────────────────
    {
        "platform": "wanted",
        "sheet_name": "Recruiting Manager",
        "base_url": "https://www.wanted.co.kr/dashboard/matchup?id=644&page={}&parentId=517&annualFrom=3&annualTo=10",
        "evaluate_fn": evaluate_recruiting_manager
    },
    # AI Contents 일시중단
    # {
    #     "platform": "wanted",
    #     "sheet_name": "Product Engineer (AI Contents)",
    #     "base_url": "https://www.wanted.co.kr/dashboard/matchup?parentId=511&page={}&annualFrom=0&annualTo=10",
    #     "evaluate_fn": evaluate_product_engineer
    # },
    # 사업 PM 일시중단
    # {
    #     "platform": "wanted",
    #     "sheet_name": "사업 PM (정부사업)",
    #     "base_url": "https://www.wanted.co.kr/dashboard/matchup-position?positionId=347296&annualFrom=2&annualTo=10&page={}",
    #     "evaluate_fn": evaluate_business_pm
    # },
    {
        "platform": "wanted",
        "sheet_name": "프로덕트 매니저 (PM)",
        "base_url": "https://www.wanted.co.kr/dashboard/matchup?id=559&page={}&annualFrom=3&annualTo=11&parentId=507",
        "evaluate_fn": evaluate_product_manager
    },

]


async def run():

    while True:

        try:

            print(f"\n===== 탐색 시작: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} =====")

            position_sheets, recommend_sheet = connect_sheets()
            embedding_store = load_embedding_store()

            today = datetime.date.today().isoformat()

            # 이전 크롬 충돌로 남은 lock 파일 정리
            for lock_file in ["SingletonLock", "SingletonCookie", "SingletonSocket"]:
                lock_path = f"/Users/cng/chrome-bot-profile/{lock_file}"
                if os.path.exists(lock_path):
                    os.remove(lock_path)
                    print(f"lock 파일 제거: {lock_file}")

            async with async_playwright() as p:

                try:
                    browser = await asyncio.wait_for(
                        p.chromium.launch_persistent_context(
                            user_data_dir="/Users/cng/chrome-bot-profile",
                            headless=False,
                            args=["--no-sandbox", "--disable-dev-shm-usage"]
                        ),
                        timeout=60
                    )
                except asyncio.TimeoutError:
                    print("Chrome 실행 타임아웃 (60초) — 재시도")
                    raise

                page = browser.pages[0] if browser.pages else await browser.new_page()

                for pos in POSITIONS:

                    position_sheet = position_sheets[pos["sheet_name"]]
                    existing_ids = load_existing_ids(position_sheet)

                    if pos["platform"] == "wanted":
                        await process_position(
                            page,
                            position_sheet,
                            recommend_sheet,
                            existing_ids,
                            today,
                            pos["sheet_name"],
                            pos["base_url"],
                            pos["evaluate_fn"],
                            embedding_store
                        )
                    else:
                        await process_position_remember(
                            page,
                            position_sheet,
                            recommend_sheet,
                            existing_ids,
                            today,
                            pos["sheet_name"],
                            pos["base_url"],
                            pos["evaluate_fn"],
                            embedding_store
                        )

                await browser.close()

            print(f"\n===== 탐색 완료. {SCAN_INTERVAL_HOURS}시간 후 재시작 =====")

        except Exception as e:

            print(f"\n메인 루프 에러: {e} — 10분 후 재시도")

            await asyncio.sleep(60 * 10)

            continue

        await asyncio.sleep(SCAN_INTERVAL_HOURS * 60 * 60)


asyncio.run(run())
