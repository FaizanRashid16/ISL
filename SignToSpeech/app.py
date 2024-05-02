import streamlit as st
import cv2
import numpy as np
import pyttsx3
import time
import tensorflow as tf

# Creating a dictionary that is later used for prediction
num_classes = {1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}
alpha_classes = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I',
                 10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'O', 16: 'P', 17: 'Q', 18: 'R',
                 19: 'S', 20: 'T', 21: 'U', 22: 'V', 23: 'W', 24: 'X', 25: 'Y', 26: 'Z'}
words_class = {1: 'All_The_Best', 2: 'Hi!!', 3: 'I_Love_you', 4: 'No', 5: 'Super!!', 6: 'Yes'}

# Necessary variables
background = None
accumulated_weight = 0.7

# Create a dictionary of custom colors
custom_colors = {
    "background": "#222222",  # Background color
    "foreground": "#FFFFFF",  # Text color
    "button_bg": "#008CBA",  # Button background color
    "button_fg": "#FFFFFF",  # Button text color
    "label_bg": "#222222",  # Label background color
    "label_fg": "#FFFFFF"  # Label text color
}

# Region of Interest (ROI) boundary
ROI_top = 100
ROI_bottom = 300
ROI_right = 300
ROI_left = 500

# Load pre-trained models
model_numbers = tf.keras.models.load_model('numbers.h5')
model_words = tf.keras.models.load_model('best.h5')
model_alphabets = tf.keras.models.load_model('alpha.h5')

# Define Streamlit app
def main():
    st.title("Sign Language Prediction")

    user_choice = st.selectbox("What do you want to predict?", ["Numbers", "Words", "Alphabets"])

    if user_choice == "Numbers":
        model = model_numbers
        text_to_speak = num_classes
    elif user_choice == "Words":
        model = model_words
        text_to_speak = words_class
    elif user_choice == "Alphabets":
        model = model_alphabets
        text_to_speak = alpha_classes

    live_prediction = st.button("Live Prediction")

    if live_prediction:
        predict_live(model, text_to_speak)

def cal_accum_avg(frame, accumulated_weight):
    global background

    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment_hand(frame, threshold=50):
    global background

    diff = cv2.absdiff(background.astype("uint8"), frame)

    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresholded, threshold1=50, threshold2=250)

    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        contour_info = [(c, cv2.contourArea(c)) for c in contours]
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        return thresholded, hand_segment_max_cont, contour_info

def predict_live(model, text_to_speak):
    cam = cv2.VideoCapture(0)
    num_frames = 0
    pred = None
    delay_time = 2

    while True:
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()
        roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

        if num_frames < 70:
            cal_accum_avg(gray_frame, accumulated_weight)
            cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            hand = segment_hand(gray_frame)
            if hand is not None:
                thresholded, hand_segment, contour_info = hand
                cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)
                cv2.imshow("Thesholded Hand Image", thresholded)
                thresholded = cv2.resize(thresholded, (64, 64))
                thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
                thresholded = np.reshape(thresholded, (1, thresholded.shape[0], thresholded.shape[1], 3))
                prev = text_to_speak[np.argmax(pred) + 1]
                pred = model.predict(thresholded)
                cv2.putText(frame_copy, text_to_speak[np.argmax(pred) + 1], (300, 45),
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 2)
                if text_to_speak[np.argmax(pred) + 1] != prev:
                    engine = pyttsx3.init()
                    engine.say(text_to_speak[np.argmax(pred) + 1])
                    engine.runAndWait()
                    prev_num_frames = num_frames
                    time.sleep(delay_time)

        cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 128, 0), 3)
        num_frames += 1
        cv2.putText(frame_copy, "Sign Language Recognition", (10, 20),
                    cv2.FONT_ITALIC, 0.5, (51, 255, 51), 1)
        cv2.imshow("Sign Detection", frame_copy)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
