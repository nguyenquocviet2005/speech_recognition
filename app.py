#!/usr/bin/env python
# coding: utf-8

"""
Real-time Vietnamese Speech Recognition App
Uses a fine-tuned Whisper model to transcribe microphone input via a Gradio interface.
"""

import gradio as gr
import torch
import numpy as np
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from noisereduce import reduce_noise

# --- Configuration ---
MODEL_PATH = "./whisper-vi-finetuned/checkpoint-7000"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLING_RATE = 16000

# --- Load Model and Processor ---
print("Loading model and processor...")
try:
    processor = WhisperProcessor.from_pretrained(MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    print("Model loaded successfully on device:", DEVICE)
except Exception as e:
    print(f"Error loading model: {e}")
    # Exit if the model can't be loaded, as the app is unusable.
    exit()

# --- Audio Preprocessing Functions (from data_preprocess.py) ---

def normalize_volume(audio, target_rms=0.1):
    """
    Standardizes the volume of the audio.
    """
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        normalized_audio = audio * (target_rms / rms)
    else:
        normalized_audio = audio
    return normalized_audio

def reduce_noise_audio(audio, sr):
    """
    Reduces background noise from the audio.
    """
    # Only reduce noise if there is substantial audio signal
    if np.sum(np.abs(audio)) > 0.01: # A small threshold
        return reduce_noise(y=audio, sr=sr, stationary=False)
    return audio

def preprocess_microphone_input(sr, audio_data):
    """
    Prepares raw microphone audio for the Whisper model.
    1. Converts to float32.
    2. Resamples to 16kHz.
    3. Normalizes volume.
    4. Reduces noise.
    """
    if audio_data is None:
        return None

    # Ensure audio is float32
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32) / 32768.0

    # Resample if necessary
    if sr != SAMPLING_RATE:
        audio_data = librosa.resample(y=audio_data, orig_sr=sr, target_sr=SAMPLING_RATE)
        sr = SAMPLING_RATE
    
    # Apply preprocessing
    processed_audio = normalize_volume(audio_data)
    processed_audio = reduce_noise_audio(processed_audio, sr)
    
    return processed_audio, sr


# --- Main Transcription Function ---

def transcribe_speech(microphone_input):
    """
    Takes microphone input, preprocesses it, and returns the transcription.
    """
    if microphone_input is None:
        return "Please record audio first to see the transcription."

    sr, audio_data = microphone_input
    
    print(f"Received audio with sample rate: {sr} and duration: {len(audio_data)/sr:.2f}s")

    # Preprocess the audio
    processed_audio, sr = preprocess_microphone_input(sr, audio_data)

    if processed_audio is None:
        return "Audio preprocessing failed."

    # Get model input features
    input_features = processor(
        processed_audio, 
        sampling_rate=sr, 
        return_tensors="pt"
    ).input_features
    
    input_features = input_features.to(DEVICE)

    # Generate token ids
    print("Generating transcription...")
    with torch.no_grad():
        generated_ids = model.generate(inputs=input_features)

    # Decode token ids to text
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("Transcription:", transcription)
    return transcription


# --- Gradio Interface ---
print("Creating Gradio interface...")

iface = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources=["microphone"], type="numpy", label="Record Vietnamese Speech"),
    outputs=gr.Textbox(label="Transcription", lines=3, placeholder="Your transcribed text will appear here..."),
    title="Vietnamese Speech Recognition with Fine-tuned Whisper",
    description="Record your voice in Vietnamese and get a live transcription. The model is a fine-tuned Whisper-small model.",
    allow_flagging="never",
    live=True # The model will transcribe as you speak
)

# --- Launch the App ---
if __name__ == "__main__":
    print("Launching app...")
    iface.launch(share=True) # Share=True creates a public link 