import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import openai
import os

# OpenAI API Key 설정
openai.api_key = os.getenv('OPENAI_API_KEY')

# TensorFlow 모델 로드 (앱 시작 시 한 번만 로드)
@st.cache_resource
def load_model():
    return tf.saved_model.load('model')

model = load_model()

# 표정 인식 함수
def predict_emotion(image, model):
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (224, 224))  # 모델의 입력 크기에 맞게 조정
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # 정규화
    predictions = model(img_array, training=False)
    class_names = ["happy", "normal", "sad", "sleepy", "surprised"]
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

# WebRTC VideoProcessor 클래스 정의
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.emotion = None
        self.frame_count = 0

    def recv(self, frame):
        self.frame_count += 1
        img = frame.to_image()  # VideoFrame을 PIL 이미지로 변환

        # 매 10번째 프레임마다 감정 예측 (프레임 처리 주기 줄이기)
        if self.frame_count % 10 == 0:
            self.emotion = predict_emotion(img, self.model)

        frame = np.array(img)  # 다시 numpy 배열로 변환
        # 감정 텍스트 추가
        if self.emotion:
            cv2.putText(frame, f"Emotion: {self.emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(frame, format="bgr24")

# Streamlit 앱 설정
st.title("실시간 감정 인식 챗봇")

# WebRTC 스트리머 시작
webrtc_ctx = webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    mode=WebRtcMode.SENDRECV
)

# 사용자 입력 받기
if webrtc_ctx.video_processor:
    user_input = st.text_input("당신: ", "")
    if user_input:
        emotion = webrtc_ctx.video_processor.emotion  # 현재 감정 가져오기
        if emotion:
            response = chat_with_gpt(emotion, user_input)
            st.write(f"ChatGPT: {response}")
        else:
            st.write("감정 인식 중입니다. 잠시만 기다려주세요.")
