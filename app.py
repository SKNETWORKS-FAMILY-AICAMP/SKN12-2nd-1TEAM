import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import warnings
import numpy as np
from datetime import datetime
from utils.stream_cache import load_data, calculate_churn_data
from utils.stream_utils import load_css, get_image_as_base64, preprocess_data
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import time
import joblib
import shap

# 한글 깨짐 수정 # 나눔고딕 설정 
plt.rc('font', family='AppleGothic')  # Mac
# plt.rc('font', family='NanumGothic') # window

# 경고 메시지 숨기기
warnings.filterwarnings('ignore', category=FutureWarning)

# 페이지 설정
st.set_page_config(
    page_title="Netflix 사용자 분석",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# CSS 로드
load_css("./styles/netflix_style.css")
load_css("./styles/styles.css")
# 데이터셋 로드 
df = load_data() # 전처리된 데이터 추가 
df_original = pd.read_csv('./data/netflix_users.csv') # 원본 데이터 추가 
# 리뷰 데이터 로드 
df_reviews = pd.read_csv('./data/netflix_reviews.csv')

# Netflix 로고 이미지 로드
netflix_logo = get_image_as_base64("./assets/netflix_logo.png")

# 사이드바 설정
st.sidebar.markdown(
    f"<div style='text-align: center;'><img src='{netflix_logo}' width='120' height='80'></div>",
    unsafe_allow_html=True
)
st.sidebar.markdown("<h2 style='text-align: center;'>Netflix 고객이탈 예측 및 분석</h2>", unsafe_allow_html=True)

# 사이드바 탭 선택
selected_tab = st.sidebar.radio(
    "분석 유형 선택",
    ["📊 데이터 분석", "🔄 고객 이탈 예측"],
    format_func=lambda x: x
)

# 제목
st.markdown(
    f"<h1><img src='{netflix_logo}' width='120' height='80' style='vertical-align: middle; margin-right: 5px;'> Netflix 사용자 분석 및 고객이탈예측</h1>",
    unsafe_allow_html=True
)



if selected_tab == "📊 데이터 분석":
    st.markdown("<div class='metric-container'><p>Netflix 고객 데이터 분석</p></div>", unsafe_allow_html=True)
    st.markdown("---")
    
    # 분석 유형 선택
    analysis_type = st.sidebar.selectbox(
        "분석 유형",
        ["전체 데이터", "국가별 분석", "디바이스별 분석",  "리뷰 분석"]
    )
    
    if analysis_type == "전체 데이터":
        st.markdown("<h2 style='text-align: center; color: #ffffff;'>Netflix 글로벌 사용자 분포</h2>", unsafe_allow_html=True)
        
        # 세계 지도 시각화
        country_stats = df_original.groupby('Country', observed=True).agg({
            'User_ID': 'count',
            'Watch_Time_Hours': 'mean',
            'monthly_income': 'mean',
            'satisfaction_score': 'mean',
        }).reset_index()

        # 국가 코드 매핑
        country_code_map = {
            'USA': 'USA', 'UK': 'GBR', 'France': 'FRA', 'Germany': 'DEU',
            'India': 'IND', 'Japan': 'JPN', 'Brazil': 'BRA', 'Canada': 'CAN',
            'Australia': 'AUS', 'Mexico': 'MEX', 'Spain': 'ESP', 'Italy': 'ITA',
            'South Korea': 'KOR', 'Russia': 'RUS', 'China': 'CHN'
        }
        country_stats['iso_alpha'] = country_stats['Country'].map(country_code_map)

        # 심플한 평면 지도와 버블 차트
        fig_map = go.Figure()

        # 기본 지도 레이어 (회색 평면 지도)
        fig_map.add_trace(go.Choropleth(
            locations=country_stats['iso_alpha'],
            z=[1] * len(country_stats),  # 모든 국가를 동일한 색상으로
            text=country_stats['Country'],
            colorscale=[[0, '#2F2F2F'], [1, '#2F2F2F']],  # 어두운 회색
            showscale=False,
            marker_line_color='#404040',
            marker_line_width=0.5,
        ))

        # 버블 레이어
        fig_map.add_trace(go.Scattergeo(
            locations=country_stats['iso_alpha'],
            mode='markers+text',
            text=country_stats['Country'],
            textposition='middle center',
            textfont=dict(size=11, color='#FFFFFF', family="Arial"),  # 텍스트 색상을 흰색으로
            marker=dict(
                size=country_stats['User_ID'] / country_stats['User_ID'].max() * 50,
                color=country_stats['Watch_Time_Hours'],
                colorscale=[
                    [0, 'rgba(229, 9, 20, 0.3)'],     # Netflix 레드 (투명도 적용)
                    [0.5, 'rgba(229, 9, 20, 0.6)'],   # Netflix 레드 (중간 투명도)
                    [1, 'rgba(229, 9, 20, 0.9)']      # Netflix 레드 (진한 색상)
                ],
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text='평균 시청 시간 (시간)',
                        font=dict(size=12, color='#FFFFFF')  # 컬러바 제목 색상
                    ),
                    thickness=10,
                    len=0.5,
                    tickfont=dict(size=10, color='#FFFFFF'),  # 컬러바 텍스트 색상
                    ticksuffix=' 시간',
                    bgcolor='rgba(0,0,0,0)',  # 컬러바 배경 투명
                    bordercolor='rgba(0,0,0,0)'  # 컬러바 테두리 투명
                ),
                line=dict(color='#404040', width=1)
            ),
            hovertemplate=
            '<b>%{customdata[0]}</b><br>' +
            '구독자 수: %{customdata[1]:,.0f}명<br>' +
            '평균 시청 시간: %{marker.color:.1f}시간<br>' +
            '<extra></extra>',
            customdata=country_stats[['Country', 'User_ID']].values
        ))

        # 지도 스타일 설정
        fig_map.update_geos(
            showcoastlines=True, coastlinecolor="#404040",
            showland=True, landcolor="#2F2F2F",
            showocean=True, oceancolor="#141414",
            showlakes=False,
            showrivers=False,
            showcountries=True, countrycolor="#404040",
            projection_type="equirectangular",
            showframe=False,
            resolution=50,
            lataxis_range=[-60, 90],  # 남극 제외
            lonaxis_range=[-180, 180],
            center=dict(lat=20, lon=0),
            projection=dict(
                scale=1.1,
                type='equirectangular'
            )
        )

        # 레이아웃 설정
        fig_map.update_layout(
            height=600,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='#141414',  # Netflix 배경색
            plot_bgcolor='#141414',
            geo=dict(
                bgcolor='#141414',  # 지도 배경색
                framecolor='#141414'  # 프레임 색상
            )
        )

        st.plotly_chart(fig_map, use_container_width=True, config={
            'displayModeBar': False,
            'scrollZoom': False  # 마우스 휠 줌 비활성화
        })

        # 주요 통계 요약
        col1, col2, col3= st.columns(3)
        with col1:
            total_users = df_original['User_ID'].count()
            st.metric("총 사용자 수", f"{total_users:,}")
        with col2:
            avg_watch = df_original['Watch_Time_Hours'].mean()
            st.metric("평균 시청 시간", f"{avg_watch:.1f}시간")
        with col3:
            avg_income = df_original['monthly_income'].mean()
            st.metric("월평균 소득", f"{avg_income:,.0f}$")
        
        # 원본 데이터 표시
        st.subheader("Netflix 사용자 데이터")
        
        # 데이터 통계 요약 표시
        st.subheader("데이터")
        st.dataframe(
            df_original.head(100),
            height=200
        )

    elif analysis_type == "국가별 분석":
        st.header("국가별 분석")
        
        # 국가 선택
        selected_countries = st.multiselect(
            "분석할 국가 선택",
            options=df['Country'].unique(),
            default=df['Country'].unique()[:5] # 기본 설정 값 
        )
        
        if selected_countries:
            # 탭 생성
            tab1, tab2, tab3, tab4 = st.tabs(["📊 구독자 추이", "💰 소득 분석", "⏱ 시청 기록", "📈 통계"])
            
            with tab1:
                # 분기별 데이터 생성
                df['Quarter'] = pd.PeriodIndex(pd.to_datetime(df['Last_Login']), freq='Q')
                quarterly_subscribers = df[df['Country'].isin(selected_countries)].groupby(['Quarter', 'Country'])['User_ID'].count().reset_index()
                
                # 누적 영역 그래프 생성
                fig_subscribers = go.Figure()
                
                for country in selected_countries:
                    country_data = quarterly_subscribers[quarterly_subscribers['Country'] == country]
                    fig_subscribers.add_trace(go.Scatter(
                        x=country_data['Quarter'].astype(str),
                        y=country_data['User_ID'],
                        name=country,
                        mode='lines',
                        stackgroup='one',
                        line=dict(width=0.5),
                        hovertemplate="<b>%{x}</b><br>" +
                                    "%{y:,.0f} subscribers<br>" +
                                    "<extra></extra>"
                    ))
                
                # 그래프 스타일 설정
                fig_subscribers.update_layout(
                    title="국가별 Netflix 구독자 수 추이",
                    title_font_size=20,
                    showlegend=True,
                    hovermode='x unified',
                    plot_bgcolor='#141414',
                    paper_bgcolor='#141414',
                    font=dict(color='white'),
                    xaxis=dict(
                        title='분기',
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(255, 255, 255, 0.1)',
                        tickfont=dict(size=12)
                    ),
                    yaxis=dict(
                        title='구독자 수',
                        showgrid=True,
                        gridwidth=1,
                        gridcolor='rgba(255, 255, 255, 0.1)',
                        tickformat=',d',
                        tickfont=dict(size=12)
                    ),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor='rgba(0,0,0,0.5)',
                        font=dict(color='white')
                    )
                )
                st.plotly_chart(fig_subscribers, use_container_width=True)
            
            with tab2:
                # 소득 분석
                st.subheader("국가별 평균 소득 분포")
                
                # 국가별 평균 소득
                income_stats = df[df['Country'].isin(selected_countries)].groupby('Country').agg({
                    'monthly_income': ['mean', 'min', 'max']
                }).reset_index()
                income_stats.columns = ['Country', 'avg_income', 'min_income', 'max_income']
                
                # 박스플롯 생성
                fig_income = go.Figure()
                
                for country in selected_countries:
                    country_income = df[df['Country'] == country]['monthly_income']
                    fig_income.add_trace(go.Box(
                        y=country_income,
                        name=country,
                        boxpoints='outliers',
                        marker_color='#E50914'
                    ))
                
                fig_income.update_layout(
                    title="국가별 소득 분포",
                    showlegend=False,
                    plot_bgcolor='#141414',
                    paper_bgcolor='#141414',
                    font=dict(color='white'),
                    yaxis_title="월 소득",
                    xaxis_title="국가",
                    yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
                )
                st.plotly_chart(fig_income, use_container_width=True)
                
                # 소득 구간별 구독자 수
                income_bins = pd.qcut(df['monthly_income'], q=5, labels=['매우 낮음', '낮음', '중간', '높음', '매우 높음'])
                df['income_level'] = income_bins
                income_dist = df[df['Country'].isin(selected_countries)].groupby(['Country', 'income_level'])['User_ID'].count().reset_index()
                
                fig_income_dist = px.bar(income_dist,
                                       x='Country',
                                       y='User_ID',
                                       color='income_level',
                                       title='소득 구간별 구독자 분포',
                                       labels={'User_ID': '구독자 수', 'income_level': '소득 수준'},
                                       color_discrete_sequence=px.colors.sequential.Reds)
                
                fig_income_dist.update_layout(
                    plot_bgcolor='#141414',
                    paper_bgcolor='#141414',
                    font=dict(color='white'),
                    yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
                )
                st.plotly_chart(fig_income_dist, use_container_width=True)
            
            with tab3:
                # 시청 기록 분석
                st.subheader("시청 시간 분석")
                
                # 국가별 평균 시청 시간
                watch_time_stats = df[df['Country'].isin(selected_countries)].groupby('Country').agg({
                    'Watch_Time_Hours': ['mean', 'sum', 'count']
                }).reset_index()
                watch_time_stats.columns = ['Country', 'avg_watch_time', 'total_watch_time', 'viewer_count']
                
                # 평균 시청 시간 그래프
                fig_watch_time = go.Figure()
                
                fig_watch_time.add_trace(go.Bar(
                    x=watch_time_stats['Country'],
                    y=watch_time_stats['avg_watch_time'],
                    name='평균 시청 시간',
                    marker_color='#E50914'
                ))
                
                fig_watch_time.update_layout(
                    title="국가별 평균 시청 시간",
                    plot_bgcolor='#141414',
                    paper_bgcolor='#141414',
                    font=dict(color='white'),
                    yaxis_title="시간",
                    xaxis_title="국가",
                    yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
                )
                st.plotly_chart(fig_watch_time, use_container_width=True)
                
                # 시청 시간대 분포
                st.subheader("시청 패턴 분석")
                viewing_pattern = df[df['Country'].isin(selected_countries)].groupby(['Country', 'Subscription_Type'])['Watch_Time_Hours'].mean().reset_index()
                
                fig_pattern = px.bar(viewing_pattern,
                                   x='Country',
                                   y='Watch_Time_Hours',
                                   color='Subscription_Type',
                                   title='구독 유형별 평균 시청 시간',
                                   barmode='group',
                                   color_discrete_sequence=px.colors.sequential.Reds)
                
                fig_pattern.update_layout(
                    plot_bgcolor='#141414',
                    paper_bgcolor='#141414',
                    font=dict(color='white'),
                    yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
                )
                st.plotly_chart(fig_pattern, use_container_width=True)
            
            with tab4:
                # 주요 통계 지표
                st.subheader("주요 통계 지표")
                
                # 국가별 주요 지표 계산
                stats = df[df['Country'].isin(selected_countries)].groupby('Country').agg({
                    'User_ID': 'count',
                    'Watch_Time_Hours': 'mean',
                    'monthly_income': 'mean',
                    'satisfaction_score': 'mean'
                }).reset_index()
                
                # 지표 표시
                for country in stats['Country']:
                    country_stats = stats[stats['Country'] == country]
                    st.markdown(f"### {country}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("총 구독자 수", f"{int(country_stats['User_ID']):,}명")
                    with col2:
                        st.metric("평균 시청 시간", f"{country_stats['Watch_Time_Hours'].values[0]:.1f}시간")
                    with col3:
                        st.metric("평균 월 소득", f"${country_stats['monthly_income'].values[0]:,.0f}")
                    with col4:
                        st.metric("만족도", f"{country_stats['satisfaction_score'].values[0]:.3f}")
                    
                    st.markdown("---")

    elif analysis_type == "디바이스별 분석":
        st.header("디바이스별 분석")
        
        # 디바이스 선택
        devices = ['Smart TV', 'Laptop', 'Mobile', 'Tablet']
        selected_devices = st.multiselect(
            "분석할 디바이스 선택",
            options=devices,
            default=devices
        )
        
        if selected_devices:
            # 시청 시간대 분석
            st.subheader("📱 디바이스별 선호 시청 시간대")
            
            # 시간대별 시청 패턴
            device_time = df[df['primary_device'].isin(selected_devices)].groupby(['primary_device', 'preferred_watching_time'])['User_ID'].count().reset_index()
            
            # 각 디바이스별 전체 사용자 수 계산
            device_totals = device_time.groupby('primary_device')['User_ID'].sum().reset_index()
            
            # 퍼센트로 변환
            device_time = device_time.merge(device_totals, on='primary_device', suffixes=('', '_total'))
            device_time['percentage'] = (device_time['User_ID'] / device_time['User_ID_total'] * 100).round(1)
            
            # 선 그래프로 시각화
            fig_time = go.Figure()
            
            for device in selected_devices:
                device_data = device_time[device_time['primary_device'] == device]
                
                fig_time.add_trace(go.Scatter(
                    x=device_data['preferred_watching_time'],
                    y=device_data['percentage'],
                    name=device,
                    mode='lines+markers',  # 선과 점을 함께 표시
                    line=dict(width=3),  # 선 굵기
                    marker=dict(size=8),  # 점 크기
                    hovertemplate="시간대: %{x}<br>" +
                                "사용자 비율: %{y:.1f}%<br>" +
                                "<extra></extra>"
                ))
            
            # 색상 맵 설정
            colors = {
                'Smart TV': '#E50914',
                'Laptop': '#B20710',
                'Mobile': '#831010',
                'Tablet': '#5C0E0E'
            }
            
            # 각 트레이스의 색상 설정
            for i, trace in enumerate(fig_time.data):
                device = selected_devices[i]
                trace.line.color = colors[device]
                trace.marker.color = colors[device]
            
            fig_time.update_layout(
                title={
                    'text': '디바이스별 시간대 선호도 분포',
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=20)
                },
                plot_bgcolor='#141414',
                paper_bgcolor='#141414',
                font=dict(color='white'),
                showlegend=True,
                legend=dict(
                    title='디바이스',
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1,
                    bgcolor='rgba(0,0,0,0.5)',
                    bordercolor='rgba(255,255,255,0.2)',
                    borderwidth=1
                ),
                xaxis=dict(
                    title='선호 시청 시간대',
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    title_font=dict(size=14),
                    tickfont=dict(size=12),
                    showgrid=True
                ),
                yaxis=dict(
                    title='사용자 비율 (%)',
                    gridcolor='rgba(255, 255, 255, 0.1)',
                    title_font=dict(size=14),
                    tickfont=dict(size=12),
                    showgrid=True,
                    range=[0, max(device_time['percentage']) * 1.1]  # 최대값에서 여유 공간 추가
                ),
                hovermode='x unified'  # 호버 모드 설정
            )
            
            # 일일 평균 시청 시간
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⏰ 일일 평균 시청 시간")
                daily_watch = df[df['primary_device'].isin(selected_devices)].groupby('primary_device')['daily_watch_hours'].mean().reset_index()
                
                fig_daily = go.Figure()
                fig_daily.add_trace(go.Bar(
                    x=daily_watch['primary_device'],
                    y=daily_watch['daily_watch_hours'],
                    marker_color='#E50914',
                    text=daily_watch['daily_watch_hours'].round(1),
                    textposition='auto',
                ))
                
                fig_daily.update_layout(
                    plot_bgcolor='#141414',
                    paper_bgcolor='#141414',
                    font=dict(color='white'),
                    title='디바이스별 일일 평균 시청 시간',
                    title_font_size=16,
                    xaxis_title="디바이스",
                    yaxis_title="시간",
                    xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
                    yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
                )
                st.plotly_chart(fig_daily, use_container_width=True)
            
            with col2:
                st.subheader("📊 누적 시청 시간")
                total_watch = df[df['primary_device'].isin(selected_devices)].groupby('primary_device')['Watch_Time_Hours'].sum().reset_index()
                
                fig_total = go.Figure()
                fig_total.add_trace(go.Bar(
                    x=total_watch['primary_device'],
                    y=total_watch['Watch_Time_Hours'],
                    marker_color='#831010',
                    text=total_watch['Watch_Time_Hours'].round(1),
                    textposition='auto',
                ))
                
                fig_total.update_layout(
                    plot_bgcolor='#141414',
                    paper_bgcolor='#141414',
                    font=dict(color='white'),
                    title='디바이스별 누적 시청 시간',
                    title_font_size=16,
                    xaxis_title="디바이스",
                    yaxis_title="시간",
                    xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
                    yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)')
                )
                st.plotly_chart(fig_total, use_container_width=True)
            
            # 디바이스별 통계
            st.subheader("📱 디바이스별 주요 통계")
            device_stats = df[df['primary_device'].isin(selected_devices)].groupby('primary_device').agg({
                'User_ID': 'count',
                'Watch_Time_Hours': 'mean', 
                'daily_watch_hours': 'mean',
                'satisfaction_score': 'mean'
            }).reset_index()
            
            for device in device_stats['primary_device']:
                device_data = device_stats[device_stats['primary_device'] == device]
                st.markdown(f"### {device}")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("사용자 수", f"{int(device_data['User_ID']):,}명")
                with col2:
                    st.metric("평균 누적 시청", f"{device_data['Watch_Time_Hours'].values[0]:.1f}시간")
                with col3:
                    st.metric("일일 평균 시청", f"{device_data['daily_watch_hours'].values[0]:.1f}시간")
                with col4:
                    st.metric("만족도", f"{device_data['satisfaction_score'].values[0]:.2f}")
                
                st.markdown("---")
    # cache 추가 
    elif analysis_type == "리뷰 분석":
        st.header("리뷰 분석")
        # 리뷰 별점 확인 
        # 리뷰 내용없는거 삭제 
        df_reviews = df_reviews[df_reviews['content'].notna()]
        review_star = df_reviews['score'].mean() # 리뷰 별점 정보 
        
        # 별점 시각화를 위한 HTML 생성
        stars_html = ""
        full_stars = int(review_star)
        has_half = (review_star - full_stars) >= 0.5
        empty_stars = 5 - full_stars - (1 if has_half else 0)
        
        # 꽉 찬 별
        stars_html += "⭐" * full_stars
        # 반 별 (필요한 경우)
        if has_half:
            stars_html += "☆"
        #빈 별
        stars_html += "☆" * empty_stars
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("리뷰 별점", f"{review_star:.2f}")

            # 세로 막대 
        with col2:
            st.markdown(f"<h3 style='margin-top: 15px;'>{stars_html}</h3>", unsafe_allow_html=True)  
            st.write(f'리뷰 개수: {len(df_reviews):,}개')
            
        # 리뷰 점수 분포 계산
        score_dist = df_reviews['score'].value_counts().sort_index()
        score_percent = (score_dist / len(df_reviews) * 100).round(1)
        
        # 수평 막대 그래프 생성
        fig_score = plt.figure(figsize=(10, 3))
        plt.style.use('dark_background')
        bars = plt.barh(score_percent.index.astype(str), score_percent.values, color='#E50914')
        plt.xlabel('퍼센트 (%)', color='white', fontsize=10)
        plt.ylabel('평점', color='white', fontsize=10)
        plt.title('리뷰 평점 분포', color='white', fontsize=12, pad=15)
        
        # 배경색 설정
        fig_score.patch.set_facecolor('#141414')
        plt.gca().set_facecolor('#141414')
        
        # 테두리 제거
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_color('#404040')
        plt.gca().spines['left'].set_color('#404040')
        
        # 격자 스타일 설정
        plt.grid(axis='x', linestyle='--', alpha=0.2, color='#404040')
        
        # 눈금 색상 설정
        plt.tick_params(axis='both', colors='white')
        
        # 막대 끝에 퍼센트 표시
        for i, v in enumerate(score_percent.values):
            plt.text(v + 0.5, i, f'{v}%', va='center', color='white', fontsize=10)
            
        plt.tight_layout()
        st.pyplot(fig_score)
            
        # 워드클라우드를 위한 리뷰 분리
        low_rating_reviews = df_reviews[df_reviews['score'] <= 3]
        high_rating_reviews = df_reviews[df_reviews['score'] >= 4]
        
        # 워드클라우드 생성을 위한 텍스트 결합
        low_rating_text = ' '.join(low_rating_reviews['content'])
        high_rating_text = ' '.join(high_rating_reviews['content'])
        
        # 워드클라우드 설정
        wordcloud_config = {
            'width': 800,
            'height': 400,
            'background_color': '#141414',
            'max_words': 200,
            'colormap': 'Purples',  # 부정적 리뷰용 컬러맵
            'font_path': '/System/Library/Fonts/Supplemental/AppleGothic.ttf'  # Mac용 폰트 경로
        }
        
        # 두 개의 컬럼 생성
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("부정적 리뷰 워드클라우드 (3점 이하)")
            # 낮은 평점 워드클라우드 (보라색 계열)
            low_wordcloud = WordCloud(**wordcloud_config).generate(low_rating_text)
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.imshow(low_wordcloud, interpolation='bilinear')
            ax1.axis('off')
            fig1.patch.set_facecolor('#141414')
            ax1.set_facecolor('#141414')
            st.pyplot(fig1)
            st.write(f"3점 이하 리뷰 수: {len(low_rating_reviews):,}개")
            
        with col2:
            st.subheader("긍정적 리뷰 워드클라우드 (4점 이상)")
            # 높은 평점 워드클라우드 (파란색 계열)
            wordcloud_config['colormap'] = 'Blues'  # 긍정적 리뷰용 컬러맵
            high_wordcloud = WordCloud(**wordcloud_config).generate(high_rating_text)
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            ax2.imshow(high_wordcloud, interpolation='bilinear')
            ax2.axis('off')
            fig2.patch.set_facecolor('#141414')
            ax2.set_facecolor('#141414')
            st.pyplot(fig2)
            st.write(f"4점 이상 리뷰 수: {len(high_rating_reviews):,}개")

else:  # 고객 이탈 예측 탭
    st.markdown("<div class='metric-container'><p>Netflix 이용자 이탈자 예측</p></div>", unsafe_allow_html=True)
    st.markdown("---")
    
    # 이탈 예측 분석 유형 선택
    churn_analysis_type = st.sidebar.selectbox(
        "이탈 분석 유형",
        ["이탈율 추이", "이탈 위험 고객 예측"]
    )
    
    if churn_analysis_type == "이탈율 추이":
        st.header("이탈율 추이 분석")
        
        # 데이터 전처리
        df['Last_Login'] = pd.to_datetime(df['Last_Login'])
        most_recent_datetime = df['Last_Login'].min()
        in_activate_date = (df['Last_Login'] - most_recent_datetime).dt.days
        in_activate_date = (in_activate_date / 7).astype(int)
        df['in_activate_date'] = in_activate_date
        
        # 주차 선택 UI 구성
        max_weeks = int(in_activate_date.max())
        weeks_options = [f"{i}주차" for i in range(max_weeks + 1)]
        
        # 현재 선택된 주차 인덱스 관리
        if 'current_week_idx' not in st.session_state:
            st.session_state.current_week_idx = 0
        
        # 버튼 클릭 핸들러
        def handle_prev_week():
            st.session_state.current_week_idx = max(0, st.session_state.current_week_idx - 1)
        
        def handle_next_week():
            st.session_state.current_week_idx = min(len(weeks_options) - 1, st.session_state.current_week_idx + 1)
        
        # UI 레이아웃
        col1, col2, col3 = st.columns([8, 1.2, 1.2])
        
        with col1:
            selected_week_str = st.selectbox(
                "분석할 주차 선택",
                weeks_options,
                index=st.session_state.current_week_idx
            )
        with col2:
            st.write("")  # 줄 간격 맞추기
            st.write("")  # 줄 간격 맞추기
            
            st.button("◀ 이전", on_click=handle_prev_week, key='prev_week', use_container_width=True)
        with col3:
            st.write("")  # 줄 간격 맞추기
            st.write("")  # 줄 간격 맞추기
            
            st.button("다음 ▶", on_click=handle_next_week, key='next_week', use_container_width=True)
        
        # 선택된 주차 값 업데이트
        selected_week = int(selected_week_str.replace('주차', ''))
        st.session_state.current_week_idx = weeks_options.index(selected_week_str)
        
        # 이탈율 계산
        total_users_start = len(df)
        active_users = len(df[df['in_activate_date'] >= selected_week])
        inactive_users = total_users_start - active_users
        churn_rate = ((total_users_start - active_users) / total_users_start) * 100
        
        # 이전 주차 데이터 계산
        prev_week = max(0, selected_week - 1)
        prev_active_users = len(df[df['in_activate_date'] >= prev_week])
        prev_inactive_users = total_users_start - prev_active_users
        prev_churn_rate = ((total_users_start - prev_active_users) / total_users_start) * 100
        
        # 주요 지표 표시
        metrics_container = st.container()
        col1, col2, col3, col4 = metrics_container.columns(4)
        
        with col1:
            st.metric("총 회원수", f"{total_users_start:,}명")
        with col2:
            active_users_delta = active_users - prev_active_users
            st.metric("활동 회원수", 
                     f"{active_users:,}명",
                     delta=f"{active_users_delta:,}명",
                     delta_color="inverse")
        with col3:
            inactive_users_delta = inactive_users - prev_inactive_users
            st.metric("비활동 회원수", 
                     f"{inactive_users:,}명",
                     delta=f"{inactive_users_delta:,}명",
                     delta_color="inverse")
        with col4:
            churn_rate_delta = churn_rate - prev_churn_rate
            st.metric("이탈율", 
                     f"{churn_rate:.1f}%",
                     delta=f"{churn_rate_delta:.1f}%",
                     delta_color="inverse")
        
        # 이탈율 추이 그래프와 활성/비활성 사용자 분포를 한 행에 배치
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("주차별 이탈율 추이")
            
            # 캐시된 함수를 사용하여 데이터 계산
            churn_df = calculate_churn_data(df, max_weeks)
            
            # 데이터 분리
            selected_mask = churn_df['주차'] <= selected_week
            selected_data = churn_df[selected_mask].copy()
            other_data = churn_df[~selected_mask].copy()
            
            # 선택된 기간의 그래프 생성 (범례 표시용)
            selected_fig = px.bar(selected_data,
                                x='주차',
                                y='이탈율',
                                color='구독 유형',
                                barmode='stack',
                                color_discrete_map={
                                    'Premium': '#E50914',
                                    'Standard': '#B20710',
                                    'Basic': '#831010'
                                })

            # 선택되지 않은 기간 (옅은 색상)
            fig_churn = px.bar(other_data, 
                              x='주차', 
                              y='이탈율',
                              color='구독 유형',
                              title=f"구독 유형별 주차별 누적 이탈율 (현재 선택: {selected_week}주차)",
                              labels={'주차': '마지막 접속으로 부터 +N주(미접속기간)', 
                                     '이탈율': '이탈율 (%)',
                                     '구독 유형': '구독 유형'},
                              barmode='stack',
                              color_discrete_map={
                                  'Premium': 'rgba(229, 9, 20, 0.2)',
                                  'Standard': 'rgba(178, 7, 16, 0.2)',
                                  'Basic': 'rgba(131, 16, 16, 0.2)'
                              })
            
            # 선택되지 않은 기간 데이터의 범례 숨기기
            for trace in fig_churn.data:
                trace.showlegend = False
            
            # 선택된 기간 데이터 추가 (진한 색상) - 범례 표시
            for trace in selected_fig.data:
                trace.showlegend = True
                fig_churn.add_trace(trace)
            
            # 그래프 레이아웃 업데이트
            fig_churn.update_layout(
                plot_bgcolor='#141414',
                paper_bgcolor='#141414',
                font_color='#ffffff',
                yaxis_gridcolor='rgba(255, 255, 255, 0.1)',
                xaxis_gridcolor='rgba(255, 255, 255, 0.1)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(color='#ffffff')
                )
            )
            
            st.plotly_chart(fig_churn, use_container_width=True)
        
        with col2:
            st.subheader("활성/비활성 사용자 분포")
            activity_dist = pd.DataFrame({
                '상태': ['활성 사용자', '비활성 사용자'],
                '사용자 수': [active_users, inactive_users]
            })
            
            # 파이 차트 생성
            fig_activity = go.Figure(data=[go.Pie(
                labels=activity_dist['상태'],
                values=activity_dist['사용자 수'],
                hole=0.7,  # 도넛 차트로 만들기
                marker=dict(colors=['#E50914', '#2d2d2d']),
                textposition='outside',
                textinfo='percent+label',
                textfont=dict(color='#ffffff', size=14)
            )])
            
            # 가운데 텍스트 추가
            total_users = activity_dist['사용자 수'].sum()
            fig_activity.add_annotation(
                text=f'총 사용자<br>{total_users:,}명',
                x=0.5, y=0.5,
                font=dict(size=16, color='#ffffff'),
                showarrow=False
            )
            
            # 차트 레이아웃 업데이트
            fig_activity.update_layout(
                plot_bgcolor='#141414',
                paper_bgcolor='#141414',
                font_color='#ffffff',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(color='#ffffff')
                )
            )
            
            st.plotly_chart(fig_activity, use_container_width=True)

    elif churn_analysis_type == "이탈 위험 고객 예측":
        st.header("고객 이탈 위험 예측")
        
        # 입력 폼 생성
        with st.form("churn_prediction_form"):
            st.subheader("고객 정보 입력")
            
            col1, col2 = st.columns(2)
            
            with col1:
                Subscription_Type = st.selectbox(
                    "구독 유형",
                    ["Basic", "Standard", "Premium"],
                    help="고객의 현재 구독 유형을 선택하세요"
                )
                Age = st.number_input(
                    "나이",
                    min_value=0,
                    max_value=150,
                    value=5,
                    step=1,
                    help="나이를 입력하세요"
                )
                
                monthly_income = st.number_input(
                    "월 소득 ($)",
                    min_value=0,
                    max_value=1000000,
                    value=0,
                    step=50,
                    help="고객의 월 소득을 입력하세요"
                )
                
                Watch_Time_Hours = st.number_input(
                    "누적 시청 시간",
                    min_value=0,
                    max_value=10000,
                    value=100,
                    step=10,
                    help="고객의 총 시청 시간을 입력하세요"
                )
                
                daily_watch_hours = st.number_input(
                    "일일 평균 시청 시간",
                    min_value=0.0,
                    max_value=24.0,
                    value=2.0,
                    step=0.5,
                    help="하루 평균 시청 시간을 입력하세요"
                )
            
            with col2:
                preferred_watching_time = st.selectbox(
                    "선호 시청 시간대",
                    ['Evening', 'Afternoon', 'Morning', 'Night'],
                    help="주로 시청하는 시간대를 선택하세요"
                )
                
                primary_device = st.selectbox(
                    "주 사용 디바이스",
                    ["Smart TV", "Laptop", "Mobile", "Tablet"],
                    help="주로 사용하는 시청 기기를 선택하세요"
                )
                
                country = st.selectbox(
                    "국가",
                    ["USA", "UK", "France", "Germany", "India", "Japan", "Brazil", 
                     "Canada", "Australia", "Mexico", "Spain", "Italy", "South Korea", 
                     "Russia", "China"],
                    help="고객의 거주 국가를 선택하세요"
                )
        
                satisfaction = st.number_input(
                    "서비스 만족도 점수",
                    min_value=0,
                    max_value=10,
                    value=5,
                    step=1,
                    help="서비스 만족도를 1~10점 사이로 선택하세요"
                )
                
                Last_Login = st.date_input(
                    "마지막 로그인 날짜",
                    help="마지막 로그인 날짜를 선택하세요"
                )
            
            
            # 추가 정보 입력
            st.subheader("추가 정보")
            col3, col4, col5 = st.columns(3)
            
            with col3:
                Favorite_Genre = st.selectbox(
                    "좋아하는 장르",
                    ['Drama', 'Sci-Fi', 'Comedy', 'Documentary', 'Romance', 'Action',
                        'Horror'],
                    help="좋아하는 장르를 선택해주세요"
                )
            
            with col4:
                profile_count = st.number_input(
                    "프로필 수 ",
                    min_value=1,
                    max_value=5,
                    value=1,
                    step=1,
                    help="생성된 프로필 수를 입력하세요"
                )
            
            with col5:
                promo_offers_used = st.number_input(
                    "프로모션 할인 ",
                    min_value=0,
                    max_value=1,
                    value=0,
                    step=1,
                    help="프로모션 할인 유무를 입력하세요"
                )
            
            # 예측 버튼
            predict_button = st.form_submit_button("이탈 위험 예측하기")
        
        # 예측 버튼이 클릭되었을 때
        if predict_button:
            try:
                # 모델 로드
                model = joblib.load('./model/XGBoost_best_model.pkl')
                # 입력 데이터 DataFrame 생성
                threshold = 0.7
                input_data = pd.DataFrame({
                    'Age': [Age],
                    'Country': [country],
                    'Subscription_Type': [Subscription_Type],
                    'Watch_Time_Hours': [Watch_Time_Hours],
                    'Favorite_Genre': [Favorite_Genre],
                    'satisfaction_score': [satisfaction],
                    'daily_watch_hours': [daily_watch_hours],
                    'primary_device': [primary_device],
                    'monthly_income': [monthly_income]  ,
                    'promo_offers_used': [promo_offers_used],
                    'profile_count': [profile_count],
                    'preferred_watching_time':[preferred_watching_time],
                    'Last_Login': [Last_Login],
                })
                input_data = preprocess_data(input_data)       
                last_login_days = input_data['Last_Login_days']
                # 예측 수행
                with st.spinner('예측 모델 분석 중...'):
                    # 프로그레스 바 표시
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                        
                    churn_prob = model.predict_proba(input_data)[0][1]
                    churn_percentage = round(churn_prob * 100, 1)
                    is_churn = churn_prob >= threshold
                
                # 예측 결과 표시
                st.subheader("예측 결과")
                
                # 이탈 여부 표시
                if is_churn:
                    st.error("⚠️ 이탈 위험이 높은 고객입니다!")
                else:
                    st.success("✅ 이탈 위험이 낮은 고객입니다!")
                
                col6, col7 = st.columns(2)
                
                with col6:
                    # 이탈 위험도를 게이지 차트로 표시
                    fig_gauge = go.Figure(go.Indicator(
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        mode = "gauge+number",
                        value = churn_percentage,
                        title = {
                            'text': "이탈 위험도",
                            'font': {'size': 28, 'color': 'white'},
                            'align': 'center'
                        },
                        number = {'suffix': "%", 'font': {'size': 50, 'color': 'white'}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickcolor': "white", 'ticksuffix': "%"},
                            'bar': {'color': "#E50914"},
                            'bgcolor': "#141414",
                            'borderwidth': 2,
                            'bordercolor': "#141414",
                            'steps': [
                                {'range': [0, 30], 'color': "rgba(229, 9, 20, 0.1)"},
                                {'range': [30, 70], 'color': "rgba(229, 9, 20, 0.3)"},
                                {'range': [70, 100], 'color': "rgba(229, 9, 20, 0.5)"}
                            ],
                            'threshold': {
                                'line': {'color': "#E50914", 'width': 4},
                                'thickness': 0.75,
                                'value': churn_percentage
                            }
                        }
                    ))
                    
                    fig_gauge.update_layout(
                        paper_bgcolor='#141414',
                        plot_bgcolor='#141414',
                        font={'color': "white", 'family': "Arial"},
                        margin=dict(t=100, b=0, l=0, r=0),
                        height=400,
                        title=None
                    )
                    
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col7:
                    # SHAP 값 계산 및 시각화
                    st.markdown("### 주요 이탈 요인 분석")
                    try:
                        # SHAP Explainer 생성
                        explainer = shap.Explainer(model)
                        # SHAP 값 계산
                        shap_values = explainer(input_data)
                        plt.rcParams['font.family'] = 'DejaVu Sans'
                        # Summary Plot
                        st.markdown("#### 전체 특성 중요도")
                        
                        # Set figure size larger
                        plt.figure(figsize=(12, 8), facecolor='#141414')
                        ax = plt.gca()
                        ax.set_facecolor('#141414')
                        
                        # Set style before creating plot
                        plt.style.use('dark_background')
                        
                        # Create SHAP plot with enhanced visibility
                        shap.summary_plot(
                            shap_values,
                            input_data,
                            plot_type="dot",
                            show=False,
                            plot_size=(12, 8),
                            color_bar_label='Feature value',
                            cmap='RdBu_r',
                            alpha=0.9,
                            max_display=10
                        )
                        
                        plt.gcf().set_facecolor('#141414')
                        plt.gca().set_facecolor('#141414')
                        
                        # Enhance grid visibility
                        ax = plt.gca()
                        ax.grid(True, color='#333333', linestyle='-', alpha=0.5, linewidth=0.5)
                        
                        # Make spines more visible
                        for spine in ax.spines.values():
                            spine.set_color('#666666')
                            spine.set_linewidth(1.5)
                            
                        # Increase text size and make more visible
                        plt.xticks(color='white', fontsize=11)
                        plt.yticks(color='white', fontsize=11)
                        ax.xaxis.label.set_color('white')
                        ax.yaxis.label.set_color('white')
                        ax.xaxis.label.set_fontsize(12)
                        ax.yaxis.label.set_fontsize(12)
                        
                        # Enhance colorbar visibility
                        cbar = plt.gcf().axes[-1]
                        cbar.set_facecolor('#141414')
                        cbar.tick_params(colors='white', labelsize=10)
                        cbar.set_title('Feature value', color='white', fontsize=12, pad=10)
                        
                        # Increase scatter plot dot sizes
                        for collection in plt.gca().collections:
                            collection.set_sizes([100])  # 점 크기 증가
                        
                        # Adjust layout with more padding
                        plt.tight_layout(pad=2.0)
                        plt.subplots_adjust(left=0.3)
                        
                        st.pyplot(plt)

                        # 주요 영향 요인 설명
                        st.markdown("#### 주요 영향 요인 설명")
                        
                        # SHAP 값을 기반으로 특성 중요도 계산 및 정렬
                        feature_importance = pd.DataFrame({
                            '특성': input_data.columns,
                            'SHAP 값': np.abs(shap_values.values[0]).mean()
                        }).sort_values('SHAP 값', ascending=False)

                        # 특성별 영향도 설명
                        feature_descriptions = {
                            'satisfaction_score': '서비스 만족도',
                            'daily_watch_hours': '일일 시청 시간',
                            'Last_Login_days': '마지막 로그인 이후 경과일',
                            'Watch_Time_Hours': '총 시청 시간',
                            'Age': '나이',
                            'preferred_watching_time': '선호 시청 시간대',
                            'monthly_income': '월 소득',
                            'promo_offers_used': '프로모션 사용 여부',
                            'primary_device': '주 사용 기기',
                            'Subscription_Type': '구독 유형'
                        }

                        # 상위 5개 특성에 대한 설명 표시
                        st.markdown("##### 🎯 상위 5개 영향 요인")
                        for idx, row in feature_importance.head().iterrows():
                            feature = row['특성']
                            impact = row['SHAP 값']
                            description = feature_descriptions.get(feature, feature)
                            
                            # SHAP 값의 부호에 따라 영향도 방향 결정
                            direction = "높임" if shap_values.values[0][idx] > 0 else "낮춤"
                            
                            st.markdown(f"""
                            **{description}** (중요도: {impact:.4f})
                            - 이 요인은 고객 이탈 가능성을 {direction}
                            - 현재 입력된 값: {input_data[feature].values[0]}
                            """)
                    except Exception as e:
                        st.error(f"SHAP 분석 중 오류가 발생했습니다: {str(e)}")

                # 맞춤형 권장사항 표시
                st.markdown("#### 고객에 맞춤 조치")
                if is_churn:
                    recommendation_count = 1
                    recommendations = []
                    
                    if float(Watch_Time_Hours) < 50:
                        recommendations.append(f"""
                        {recommendation_count}. 🎯 **맞춤형 컨텐츠 추천 강화**
                        - 고객의 시청 기록을 기반으로 더 정확한 추천 제공
                        - 새로운 장르의 컨텐츠 노출 확대
                        """)
                        recommendation_count += 1
                    
                    if Subscription_Type == "Premium":
                        recommendations.append(f"""
                        {recommendation_count}. 💰 **특별 할인 프로모션 제공**
                        - 다음 달 구독료 20% 할인 쿠폰 제공
                        - 연간 구독 전환 시 추가 할인 제공
                        """)
                        recommendation_count += 1
                    else:
                        recommendations.append(f"""
                        {recommendation_count}. 💰 **업그레이드 프로모션 제공**
                           - 프리미엄 구독으로 업그레이드 시 첫 달 50% 할인
                           - 화질 및 동시 시청 장점 강조
                        """)
                        recommendation_count += 1
                    
                    if int(profile_count) == 1:
                        recommendations.append(f"""
                        {recommendation_count}. 👥 **프로필 활용도 증대**
                           - 가족 프로필 생성 안내
                           - 프로필별 맞춤 설정 가이드 제공
                        """)
                        recommendation_count += 1
                    
                    recommendations.append(f"""
                    {recommendation_count}. 🤝 **개인화된 서비스 제공**
                       - 선호 장르 기반 신작 알림 서비스
                       - 맞춤형 시청 가이드 제공
                    """)
                    
                    # 모든 권장사항 표시
                    for rec in recommendations:
                        st.markdown(rec)
                else:
                    st.markdown("# 추천 권장사항이 없습니다.")
            
            except Exception as e:
                st.error(f"예측 중 오류가 발생했습니다: {str(e)}")
                st.info("입력값을 확인하고 다시 시도해주세요.")
