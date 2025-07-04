#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import librosa
import numpy as np
from noisereduce import reduce_noise
import soundfile as sf


data_dir = "data"
audio_dir = os.path.join(data_dir, "mp3")
transcript_file = os.path.join(data_dir, "transcriptAll.txt")
output_dir = "processed_audio"
os.makedirs(output_dir, exist_ok=True)

def normalize_text(text):
    text = text.strip().lower()
    text = re.sub(r'[^\w\sàáảãạăắằẵẳặâầấậẫẩđèéẹẻẽêềếểễệìíịỉĩòóọỏõôồốổỗộơờớởỡợùúụủũưừứửữựỳýỵỷỹ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Hàm chuẩn hóa âm lượng (RMS normalization)
def normalize_volume(audio, target_rms=0.1):
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:  # Tránh chia cho 0
        normalized_audio = audio * (target_rms / rms)
    else:
        normalized_audio = audio
    return normalized_audio

def reduce_noise_audio(audio, sr):
    if np.sum(np.abs(audio)) == 0:
        return audio
    return reduce_noise(y=audio, sr=sr)

# Hàm xử lý file âm thanh (normalize + noise reduction)
def process_audio(input_path, output_path, start_time, end_time, sr=16000):
    try:
        duration = end_time - start_time
        if duration <= 0:
            print(f"Bỏ qua vì thời lượng không hợp lệ: {input_path} ({start_time}-{end_time})")
            return False
        audio, _ = librosa.load(input_path, sr=sr, offset=start_time, duration=duration)
        # Áp dụng noise reduction
        audio = reduce_noise_audio(audio, sr)
        # Áp dụng volume normalization
        audio = normalize_volume(audio)
        sf.write(output_path, audio, sr)
        return True
    except Exception as e:
        print(f"Lỗi xử lý {input_path}: {e}")
        return False

entries = []
with open(transcript_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) == 3:
            filename, transcript, time_range = parts
            audio_path = os.path.join(audio_dir, filename)
            if os.path.exists(audio_path):
                time_parts = re.split(r"[-\s]+", time_range.strip())
                if len(time_parts) >= 2:
                    try:
                        start = float(time_parts[0])
                        end = float(time_parts[1])
                        output_audio_path = os.path.join(output_dir, filename.replace('.mp3', '.wav'))
                        if process_audio(audio_path, output_audio_path, start, end):
                            entries.append({
                                "path": output_audio_path,
                                "transcription": normalize_text(transcript),
                                "start": start,
                                "end": end
                            })
                    except ValueError:
                        print(f"Không thể convert thời gian: {time_range}")
                else:
                    print(f"Không đủ start/end: {time_range}")
        else:
            print(f"Dòng lỗi định dạng: {line.strip()}")

df = pd.DataFrame(entries)
print(df.head())

df_train, df_temp = train_test_split(df, test_size=0.2, random_state=42)
df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

df_train.to_csv("fpt_train.csv", index=False, encoding="utf-8")
df_val.to_csv("fpt_val.csv", index=False, encoding="utf-8")
df_test.to_csv("fpt_test.csv", index=False, encoding="utf-8")

print(f"Đã lưu: fpt_train.csv ({len(df_train)} mẫu), fpt_val.csv ({len(df_val)} mẫu), fpt_test.csv ({len(df_test)} mẫu)")


# In[5]:


import librosa

audio_path = df["path"].iloc[5]
y, sr = librosa.load(audio_path, sr=None)  
print(f"Sampling rate: {sr} Hz")


# In[ ]:




