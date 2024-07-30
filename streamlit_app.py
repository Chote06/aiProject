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

# Streamlit 카메라 입력
image = st.camera_input("웹캠을 사용하여 사진을 찍어주세요")

if image is not None:
    # 이미지를 화면에 표시
    st.image(image, caption="Captured Image", use_column_width=True)
    
    # 표정 인식
    emotion = predict_emotion(image, model)
    st.write(f"감지된 감정: {emotion}")
    
    # 사용자 입력 받기
    user_input = st.text_input("당신: ", "")
    if user_input:
        response = chat_with_gpt(emotion, user_input)
        st.write(f"ChatGPT: {response}")
