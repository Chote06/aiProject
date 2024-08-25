import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import openai
import os

# OpenAI API Key 설정
openai.api_key = os.getenv('OPENAI_API_KEY')

# TensorFlow SavedModel 로드
model = tf.saved_model.load('model')

# Streamlit 앱 설정
st.title("감정 챗봇")
st.header("감정을 인식하여 대화하는 AI")

# 표정 인식 함수
def predict_emotion(image, model):
    img_array = cv2.resize(np.array(image), (224, 224))  # 모델의 입력 크기에 맞게 조정
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # 정규화
    # 모델 예측
    predictions = model(img_array, training=False)
    class_names = ["happy", "normal", "sad", "sleepy", "surprised"]  # 모델에 맞는 클래스 이름 설정
    return class_names[np.argmax(predictions)]

# OpenAI ChatGPT 호출 함수
def chat_with_gpt(emotion, user_input):
    prompt = f"너는 사용자의 기분에 따라 그들과 상호작용하는 AI야. 사용자는 {emotion}한 감정을 느끼고 있어. 사용자는 \"{user_input}\"라 말하고 있어. 상황에 맞춰 대답해."
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# 웹캠에서 실시간 비디오 피드를 가져오기
cap = cv2.VideoCapture(0)  # 웹캠 장치 ID는 일반적으로 0입니다.

# 스트림이 열려있는 동안 계속해서 프레임을 가져옴
if cap.isOpened():
    stframe = st.empty()  # Streamlit에서 이미지를 표시할 공간 확보
    
    while True:
        ret, frame = cap.read()  # 프레임 읽기
        if not ret:
            st.write("카메라에서 프레임을 읽을 수 없습니다.")
            break
        
        # 이미지 처리를 위한 OpenCV BGR -> RGB 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_frame)  # PIL 이미지로 변환
        
        # 표정 인식
        emotion = predict_emotion(img_pil, model)
        
        # 화면에 인식된 감정 출력
        cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Streamlit을 통해 화면에 프레임 표시
        stframe.image(frame, channels="BGR")
        
        # 사용자 입력 받기
        user_input = st.text_input("당신: ", "")
        if user_input:
            response = chat_with_gpt(emotion, user_input)
            st.write(f"ChatGPT: {response}")

    cap.release()
else:
    st.write("카메라를 열 수 없습니다.")
