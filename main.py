# my-data
import streamlit as st
import pandas as pd

# 페이지 설정
st.set_page_config(
    page_title="서울 연평균 기온 변화",
    page_icon="🌡️",
    layout="wide"
)

# 데이터 주소
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"


@st.cache_data
def load_data():
    """서울 기온 데이터를 불러와 연평균 기온을 계산합니다."""
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 날짜를 날짜 형식으로 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온을 숫자로 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 날짜 또는 평균기온이 없는 행 제거
    df = df.dropna(subset=["날짜", "평균기온"])

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 연도별 평균기온 계산
    yearly = (
        df.groupby("연도")["평균기온"]
        .mean()
        .reset_index()
        .sort_values("연도")
    )

    return yearly


# 데이터 불러오기
data = load_data()

# 제목
st.title("🌡️ 서울의 100년 연평균 기온 변화")
st.caption("서울의 일평균 기온 데이터를 연도별로 평균하여 나타낸 그래프입니다.")

# 분석 기간 표시
if not data.empty:
    start_year = int(data["연도"].min())
    end_year = int(data["연도"].max())

    st.info(
        f"분석 기간: **{start_year}년 ~ {end_year}년** "
        f"({end_year - start_year + 1}년)"
    )

# 그래프
chart_data = data.set_index("연도")

st.line_chart(
    chart_data,
    y="평균기온",
    x_label="연도",
    y_label="연평균 기온 (℃)",
    height=500
)

# 간단한 요약
if len(data) >= 2:
    first_year = int(data.iloc[0]["연도"])
    last_year = int(data.iloc[-1]["연도"])
    first_temp = data.iloc[0]["평균기온"]
    last_temp = data.iloc[-1]["평균기온"]
    change = last_temp - first_temp

    st.subheader("📊 변화 요약")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "시작 연도 평균기온",
            f"{first_temp:.1f} ℃",
            f"{first_year}년"
        )

    with col2:
        st.metric(
            "최근 연도 평균기온",
            f"{last_temp:.1f} ℃",
            f"{last_year}년"
        )

    with col3:
        st.metric(
            "전체 기간 변화",
            f"{change:+.1f} ℃"
        )

st.caption(
    "출처: 기상청 서울 기온 관측 데이터 (GitHub modudata)"
)
