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
st.caption("서울의 연평균기온 데이터를 이용해 연도에 따른 기온 변화를 살펴보고 회귀선으로 예상 기온을 확인합니다.")


# ---------------------------------
# 데이터 불러오기
# ---------------------------------
URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(URL, encoding="utf-8-sig")

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 숫자형 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 결측값 제거
    df = df.dropna(subset=["날짜", "평균기온"])

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
# 1. 2025년 이후 제외
# 2. 관측일수가 300일 미만인 해 제외
# ---------------------------------
yearly = yearly[
    (yearly["연도"] <= 2025) &
    (yearly["관측일수"] >= 300)
].copy()

yearly = yearly.sort_values("연도").reset_index(drop=True)


# ---------------------------------
# 회귀용 데이터
# 독립변수 = 1908년부터 지난 연수
# ---------------------------------
yearly["지난연수"] = yearly["연도"] - 1908

x = yearly["지난연수"].to_numpy()
y = yearly["평균기온"].to_numpy()


# ---------------------------------
# 선형 회귀
# ---------------------------------
slope, intercept = np.polyfit(x, y, 1)

yearly["회귀예상기온"] = (
    intercept + slope * yearly["지난연수"]
)

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]


# ---------------------------------
# 회귀선에 사용할 연도 범위
# 1908년부터 2025년까지
# ---------------------------------
regression_years = np.arange(
    1908,
    2026
)

regression_x = regression_years - 1908

regression_temperature = (
    intercept + slope * regression_x
)

regression_df = pd.DataFrame({
    "연도": regression_years,
    "회귀예상기온": regression_temperature
})


# ---------------------------------
# 회귀선 정보
# ---------------------------------
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
number_of_years = len(yearly)


# ---------------------------------
# 화면에 회귀 정보 표시
# ---------------------------------
st.subheader("📊 회귀 분석 정보")

col1, col2, col3, col4 = st.columns(4)

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

with col4:
    st.metric(
        "상관계수",
        f"{correlation:.3f}"
    )


# ---------------------------------
# 산점도 + 회귀선
# ---------------------------------
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

# 회귀선 추가
fig.add_scatter(
    x=regression_df["연도"],
    y=regression_df["회귀예상기온"],
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


# ---------------------------------
# 예상 기온 슬라이더
# ---------------------------------
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
    intercept + slope * selected_x
)


# ---------------------------------
# 선택한 연도의 예상 기온
# ---------------------------------
st.markdown(
    f"### {selected_year}년 예상 연평균기온"
)

st.metric(
    label="회귀선으로 계산한 예상 기온",
    value=f"{predicted_temperature:.2f} ℃"
)


# ---------------------------------
# 선택 연도 설명
# ---------------------------------
if selected_year < start_year:
    st.info(
        f"{selected_year}년은 실제 회귀에 사용한 데이터의 시작 연도 "
        f"({start_year}년)보다 이전이므로 회귀선을 이용한 예상값입니다."
    )

elif selected_year > end_year:
    st.info(
        f"{selected_year}년은 실제 회귀에 사용한 데이터의 마지막 연도 "
        f"({end_year}년) 이후이므로 회귀선을 이용한 예상값입니다."
    )

else:
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

        difference = predicted_temperature - actual_temperature

        st.write(
            f"회귀선 예상값과 실제값의 차이: "
            f"**{difference:+.2f} ℃**"
        )


# ---------------------------------
# 안내
# ---------------------------------
st.divider()

st.caption(
    "※ 회귀선은 1908년을 0으로 하는 '지난 연수'를 독립 변수로 사용했습니다. "
    "2025년까지의 데이터 중 관측일수가 300일 이상인 연도만 회귀 분석에 사용했습니다."
)
