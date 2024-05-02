import streamlit as st
import subprocess
import os

# Function for Sign to Speech
def sign_to_speech():
    result_label.text("Sign to Speech selected")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    speech_to_sign_dir = os.path.join(script_dir, "SignToSpeech")
    os.chdir(speech_to_sign_dir)
    subprocess.Popen(["streamlit", "run", "app.py"])

# Function for Speech to Sign
def speech_to_sign():
    result_label.text("Speech to Sign selected")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    speech_to_sign_dir = os.path.join(script_dir, "SpeechToSign")
    os.chdir(speech_to_sign_dir)
    subprocess.Popen(["streamlit", "run", "app.py"])

st.title("Sign to Speech / Speech to Sign")
st.markdown("Select an option:")

# Display the selected option
result_label = st.empty()

# Create buttons for the two choices
sign_to_speech_button = st.button("Sign to Speech", on_click=sign_to_speech)
speech_to_sign_button = st.button("Speech to Sign", on_click=speech_to_sign)
