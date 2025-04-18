import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    df = pd.read_csv('./data/netflix_users.csv')
    
    # 기본 데이터 전처리
    # 마지막 접속일이 현재와 가까울수록 활성 사용자로 처리
    current_date = pd.Timestamp.now()
    df['days_active'] = df['Last_Login'].apply(lambda x: (pd.Timestamp.now() - pd.Timestamp(x)).days)
    df['days_since_last'] = 365 - df['days_active']  # 1년을 기준으로 반전
    df['activation_type'] = df['days_since_last'].apply(lambda x: 'active' if x >= 353 else 'inactive')  # 12일 이내 접속한 사용자를 활성으로
    
    # 1. 시청 패턴 관련 컬럼
    df['daily_watch_time'] = df['daily_watch_hours']  # 이미 일일 시청 시간이 있음
    
    # 2. 재무 관련 컬럼
    subscription_prices = {
        'Basic': 9.99,
        'Standard': 13.99,
        'Premium': 17.99
    }
    df['monthly_revenue'] = df['Subscription_Type'].map(subscription_prices)
    df['revenue_per_watch_hour'] = df['monthly_revenue'] / df['Watch_Time_Hours']    
    return df


@st.cache_data
def calculate_churn_data(df, max_weeks):
    """주차별 구독 유형별 이탈율 계산 - 캐싱 적용"""
    churn_data = []
    total_users = len(df)
    
    for week in range(max_weeks + 1):
        week_data = []
        for sub_type in df['Subscription_Type'].unique():
            sub_users = df[df['Subscription_Type'] == sub_type]
            total_sub_users = len(sub_users)
            active_sub_users = len(sub_users[sub_users['in_activate_date'] >= week])
            inactive_sub_users = total_sub_users - active_sub_users
            # 전체 사용자 대비 해당 구독의 이탈 사용자 비율 계산
            week_churn_rate = (inactive_sub_users / total_users) * 100
            week_data.append({
                '주차': week,
                '구독 유형': sub_type,
                '이탈율': week_churn_rate
            })
        churn_data.extend(week_data)
    
    return pd.DataFrame(churn_data)
