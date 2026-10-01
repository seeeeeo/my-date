import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# ---------------------------------
# 기본 설정
# ---------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.caption(
    "서울의 연평균기온 데이터를 이용해 연도에 따른 기온 변화를 살펴보고 "
    "회귀선으로 예상 기온을 확인합니다."
)


# ---------------------------------
# 데이터 불러오기
# ---------------------------------
URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(URL, encoding="utf-8-sig")

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["날짜", "평균기온"]
    )

    return df


df = load_data()


# ---------------------------------
# 연도별 평균기온 계산
# ---------------------------------
df["연도"] = df["날짜"].dt.year

yearly = (
    df.groupby("연도")
    .agg(
        평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)


# ---------------------------------
# 분석 대상 기간
# 2025년 이후 제외
# 관측일수 300일 미만 제외
# ---------------------------------
yearly = yearly[
    (yearly["연도"] <= 2025)
    & (yearly["관측일수"] >= 300)
].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)


# ---------------------------------
# 독립변수
# 1908년부터 지난 연수
# ---------------------------------
yearly["지난연수"] = yearly["연도"] - 1908


# =================================================
# 1. 전체 기간 회귀
# =================================================

x_all = yearly["지난연수"].to_numpy()
y_all = yearly["평균기온"].to_numpy()

slope_all, intercept_all = np.polyfit(
    x_all,
    y_all,
    1
)

# 1년에 몇 도 변화하는가
slope_all_per_year = slope_all

# 100년에 몇 도 변화하는가
slope_all_100 = slope_all_per_year * 100

# 상관계수
correlation = np.corrcoef(
    x_all,
    y_all
)[0, 1]


# =================================================
# 2. 최근 20년 회귀
# =================================================

latest_year = int(yearly["연도"].max())

recent_start_year = latest_year - 19

recent20 = yearly[
    yearly["연도"] >= recent_start_year
].copy()

x_recent = recent20["지난연수"].to_numpy()
y_recent = recent20["평균기온"].to_numpy()

slope_recent, intercept_recent = np.polyfit(
    x_recent,
    y_recent,
    1
)

slope_recent_100 = slope_recent * 100


# ---------------------------------
# 회귀선 데이터
# ---------------------------------

regression_years = np.arange(
    start=1908,
    stop=2026
)

regression_x = regression_years - 1908


# 전체 기간 회귀선
regression_all = (
    intercept_all
    + slope_all * regression_x
)


# 최근 20년 회귀선
regression_recent = (
    intercept_recent
    + slope_recent * regression_x
)


# =================================================
# 회귀 정보
# =================================================

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
number_of_years = len(yearly)


st.subheader("📊 회귀 분석 정보")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "회귀에 사용한 연도 수",
        f"{number_of_years}년"
    )

with col2:
    st.metric(
        "시작 연도",
        f"{start_year}년"
    )

with col3:
    st.metric(
        "끝 연도",
        f"{end_year}년"
    )


# =================================================
# 100년에 몇 도 오르는가?
# =================================================

st.subheader("🌡️ 100년에 몇 도 오르는가?")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "전체 기간",
        f"{slope_all_100:+.2f} ℃",
        help="전체 회귀 기간을 기준으로 100년 동안 변하는 연평균기온"
    )

with col2:
    st.metric(
        f"최근 20년 ({recent_start_year}~{latest_year})",
        f"{slope_recent_100:+.2f} ℃",
        help="최근 20년의 연평균기온만 사용해 계산한 100년당 변화량"
    )


# =================================================
# 산점도 + 회귀선
# =================================================

st.subheader("📈 연도별 평균기온과 회귀선")

fig = px.scatter(
    yearly,
    x="연도",
    y="평균기온",
    hover_data={
        "연도": True,
        "평균기온": ":.2f",
        "관측일수": True
    },
    labels={
        "연도": "연도",
        "평균기온": "연평균기온 (℃)",
        "관측일수": "관측일수"
    },
    title="서울 연도별 평균기온"
)


# 전체 기간 회귀선
fig.add_scatter(
    x=regression_years,
    y=regression_all,
    mode="lines",
    name="전체 기간 회귀선",
    hovertemplate=(
        "연도: %{x}<br>"
        "예상기온: %{y:.2f}℃"
        "<extra></extra>"
    )
)


# 최근 20년 회귀선
fig.add_scatter(
    x=regression_years,
    y=regression_recent,
    mode="lines",
    name="최근 20년 회귀선",
    hovertemplate=(
        "연도: %{x}<br>"
        "최근 20년 회귀 예상기온: %{y:.2f}℃"
        "<extra></extra>"
    )
)


fig.update_layout(
    xaxis=dict(
        tickmode="linear",
        dtick=10,
        title="연도"
    ),
    yaxis_title="연평균기온 (℃)",
    hovermode="closest"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# =================================================
# 최근 20년 비교 설명
# =================================================

st.subheader("🔎 전체 기간과 최근 20년 비교")

if slope_recent_100 > slope_all_100:
    comparison_text = (
        "최근 20년의 100년당 기온 변화량이 "
        "전체 기간의 변화량보다 큽니다."
    )
elif slope_recent_100 < slope_all_100:
    comparison_text = (
        "최근 20년의 100년당 기온 변화량이 "
        "전체 기간의 변화량보다 작습니다."
    )
else:
    comparison_text = (
        "최근 20년과 전체 기간의 100년당 "
        "기온 변화량이 같습니다."
    )

st.info(comparison_text)


# =================================================
# 예상 기온 슬라이더
# =================================================

st.subheader("🌡️ 원하는 연도의 예상 기온")

selected_year = st.slider(
    "연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)


# 1908년부터 지난 연수
selected_x = selected_year - 1908

predicted_temperature = (
    intercept_all
    + slope_all * selected_x
)


st.markdown(
    f"### {selected_year}년 예상 연평균기온"
)

st.metric(
    label="전체 기간 회귀선으로 계산한 예상 기온",
    value=f"{predicted_temperature:.2f} ℃"
)


# ---------------------------------
# 실제 데이터가 있는 연도라면 비교
# ---------------------------------

actual = yearly.loc[
    yearly["연도"] == selected_year,
    "평균기온"
]

if not actual.empty:

    actual_temperature = actual.iloc[0]

    st.write(
        f"실제 {selected_year}년 연평균기온: "
        f"**{actual_temperature:.2f} ℃**"
    )

    difference = (
        predicted_temperature
        - actual_temperature
    )

    st.write(
        f"회귀선 예상값과 실제값의 차이: "
        f"**{difference:+.2f} ℃**"
    )

else:

    st.info(
        f"{selected_year}년은 실제 회귀 분석 데이터에 없으므로 "
        "회귀선을 이용한 예상값만 표시합니다."
    )


# ---------------------------------
# 안내
# ---------------------------------

st.divider()

st.caption(
    "※ 2025년까지의 데이터 중 관측일수가 300일 이상인 연도만 분석에 사용했습니다. "
    "회귀의 독립변수는 1908년부터 지난 연수입니다."
)
