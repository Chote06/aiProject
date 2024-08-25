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
        self.last_frame = None

    def recv(self, frame):
        img = frame.to_image()  # VideoFrame을 PIL 이미지로 변환
        self.last_frame = img  # 최신 프레임 저장
        return av.VideoFrame.from_image(img)  # 프레임을 그대로 반환 (화면 출력은 안함)

    def get_emotion(self):
        if self.last_frame is not None:
            return predict_emotion(self.last_frame, self.model)
        return None

# Streamlit 앱 설정
st.title("실시간 감정 인식 챗봇")

# WebRTC 스트리머 시작 (비디오 출력 없이 감정만 인식)
webrtc_ctx = webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    mode=WebRtcMode.SENDONLY
)

# 사용자 입력 받기 및 처리
if webrtc_ctx.video_processor:
    user_input = st.text_input("당신: ", "")
    if user_input:
        # 웹캠 프레임에서 감정 예측
        emotion = webrtc_ctx.video_processor.get_emotion()
        if emotion:
            response = chat_with_gpt(emotion, user_input)
            st.write(f"ChatGPT: {response}")
        else:
            st.write("웹캠에서 프레임을 캡처할 수 없습니다. 다시 시도해주세요.")
