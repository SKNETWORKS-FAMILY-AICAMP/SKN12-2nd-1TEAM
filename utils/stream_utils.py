import streamlit as st
import pathlib
import base64
from PIL import Image
import io
import pandas as pd 
import joblib
import datetime

def preprocess_data(df):
    try:
        current_date = datetime.datetime(2025,3,8)
        current_date = pd.to_datetime(current_date)
        label_encoder = joblib.load('./model/label_encoder.pkl')
        # 날짜 처리
        df['Last_Login'] = pd.to_datetime(df['Last_Login'])
        df['Last_Login_days'] = (current_date - df['Last_Login']).dt.days
        # 음수 값 처리
        df['Last_Login_days'] = df['Last_Login_days'].apply(lambda x: 0 if x < 0 else x)
        # Last_Login 컬럼 제거
        df = df.drop(columns=['Last_Login'])
        
        # 범주형 변수 선택 및 인코딩
        categorical_columns = df.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            df[column] = label_encoder.fit_transform(df[column])
        
        return df
        
    except Exception as e:
        print(f"전처리 중 오류 발생: {str(e)}")
        raise e


# CSS 파일 로드
def load_css(path):
    css_file = pathlib.Path(path)
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# 이미지를 base64로 인코딩하는 함수
def get_image_as_base64(file_path):
    try:
        img = Image.open(file_path)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
