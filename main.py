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
URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(
        URL,
        encoding="utf-8-sig"
    )

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
# 기준 기간 적용
# 2025년 이후 제외
# 관측일수 300일 미만 제외
# ---------------------------------
yearly = yearly[
    (yearly["연도"] <= 2025)
    & (yearly["관측일수"] >= 300)
].copy()

yearly = yearly.sort_values(
    "연도"
).reset_index(drop=True)


# ---------------------------------
# 회귀용 데이터
# 독립변수 = 1908년부터 지난 연수
# ---------------------------------
yearly["지난연수"] = (
    yearly["연도"] - 1908
)

x = yearly["지난연수"].to_numpy()
y = yearly["평균기온"].to_numpy()


# ---------------------------------
# 전체 기간 선형 회귀
# ---------------------------------
slope, intercept = np.polyfit(
    x,
    y,
    1
)


yearly["회귀예상기온"] = (
    intercept
    + slope * yearly["지난연수"]
)


# ---------------------------------
# 상관계수
# ---------------------------------
correlation = np.corrcoef(
    x,
    y
)[0, 1]


# ---------------------------------
# 100년당 기온 변화량
# ---------------------------------
slope_100 = slope * 100


# ---------------------------------
# 회귀선
# ---------------------------------
regression_years = np.arange(
    1908,
    2026
)

regression_x = (
    regression_years - 1908
)

regression_temperature = (
    intercept
    + slope * regression_x
)


# ---------------------------------
# 회귀 분석 정보
# ---------------------------------
start_year = int(
    yearly["연도"].min()
)

end_year = int(
    yearly["연도"].max()
)

number_of_years = len(
    yearly
)


# =================================================
# 회귀 분석 정보
# =================================================

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
# 상관계수
# =================================================

st.subheader("🔗 연도와 평균기온의 상관관계")

st.metric(
    "상관계수",
    f"{correlation:.3f}"
)

st.caption(
    "상관계수는 연도와 연평균기온이 함께 변하는 정도를 나타냅니다."
)


# =================================================
# 100년에 몇 도 오르는가?
# =================================================

st.subheader("🌡️ 100년에 몇 도 오르는가?")

st.metric(
    "전체 기간의 기온 변화량",
    f"{slope_100:+.2f} ℃ / 100년"
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


# ---------------------------------
# 회귀선 추가
# ---------------------------------
fig.add_scatter(
    x=regression_years,
    y=regression_temperature,
    mode="lines",
    name="회귀선",
    hovertemplate=(
        "연도: %{x}<br>"
        "회귀 예상기온: %{y:.2f}℃"
        "<extra></extra>"
    )
)


fig.update_layout(
    xaxis=dict(
        title="연도",
        tickmode="linear",
        dtick=10
    ),
    yaxis=dict(
        title="연평균기온 (℃)"
    ),
    hovermode="closest"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# =================================================
# 최근 20년 회귀
# =================================================

latest_year = int(
    yearly["연도"].max()
)

recent_start_year = (
    latest_year - 19
)


recent20 = yearly[
    yearly["연도"] >= recent_start_year
].copy()


recent_x = recent20["지난연수"].to_numpy()
recent_y = recent20["평균기온"].to_numpy()


recent_slope, recent_intercept = np.polyfit(
    recent_x,
    recent_y,
    1
)


recent_slope_100 = (
    recent_slope * 100
)


# =================================================
# 전체 기간 vs 최근 20년
# =================================================

st.subheader("📊 전체 기간과 최근 20년 비교")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        f"전체 기간 ({start_year}~{end_year})",
        f"{slope_100:+.2f} ℃ / 100년"
    )

with col2:
    st.metric(
        f"최근 20년 ({recent_start_year}~{latest_year})",
        f"{recent_slope_100:+.2f} ℃ / 100년"
    )


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


# ---------------------------------
# 선택한 연도의 예상 기온
# ---------------------------------
selected_x = (
    selected_year - 1908
)

predicted_temperature = (
    intercept
    + slope * selected_x
)


st.markdown(
    f"### {selected_year}년 예상 연평균기온"
)

st.metric(
    "회귀선으로 계산한 예상 기온",
    f"{predicted_temperature:.2f} ℃"
)


# ---------------------------------
# 실제 데이터가 있으면 비교
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
        f"{selected_year}년은 실제 분석 데이터에 없으므로 "
        "회귀선을 이용한 예상값만 표시합니다."
    )


# =================================================
# 안내
# =================================================

st.divider()

st.caption(
    "※ 2025년까지의 데이터 중 관측일수가 300일 이상인 연도만 "
    "회귀 분석에 사용했습니다."
)

st.caption(
    "※ 회귀의 독립변수는 1908년부터 지난 연수입니다."
)

st.caption(
    "※ 상관계수는 연도와 연평균기온 사이의 피어슨 상관계수입니다."
)
