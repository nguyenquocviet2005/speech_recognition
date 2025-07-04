# VietASR Whisper Fine-tuning Project

A Vietnamese Automatic Speech Recognition (ASR) system built by fine-tuning OpenAI's Whisper model on Vietnamese speech data. This project includes data preprocessing, model training, evaluation, and a real-time Gradio web interface.

## 🎯 Project Overview

This project fine-tunes the Whisper-small model specifically for Vietnamese speech recognition, achieving improved performance on Vietnamese audio compared to the base multilingual model. The system includes comprehensive preprocessing, evaluation metrics, and a user-friendly web interface.

## 📁 Project Structure

```
VietASR_whisper_finetune/
├── app.py                              # Gradio web interface for real-time ASR
├── run_app.sh                          # Script to launch the web app
├── finetune_whisper_vietnamese.py     # Main training script
├── finetune_whisper_clean.py          # Clean version of training script
├── finetune_whisper_vietnamese.ipynb  # Jupyter notebook for training
├── data_preprocess.py                  # Audio preprocessing utilities
├── data_preprocess.ipynb              # Data preprocessing notebook
├── evaluate_model.py                  # Model evaluation script
├── run_evaluation.sh                  # Script to run evaluation
├── evaluation_results.txt             # Evaluation metrics results
├── requirements.txt                   # Python dependencies
├── test_audio.wav                     # Sample test audio
├── fpt_train.csv                      # Training dataset metadata
├── fpt_val.csv                        # Validation dataset metadata
├── fpt_test.csv                       # Test dataset metadata
├── whisper-vi-finetuned/              # Fine-tuned model directory
├── processed_audio/                   # Preprocessed audio files
└── data/                              # Raw audio data
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd VietASR_whisper_finetune

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Web Interface

```bash
# Make the script executable
chmod +x run_app.sh

# Launch the Gradio app
./run_app.sh
```

Or run directly:
```bash
python app.py
```

### 3. Model Evaluation

```bash
# Run evaluation on test set
chmod +x run_evaluation.sh
./run_evaluation.sh
```

Or run directly:
```bash
python evaluate_model.py
```

## 🔧 Features

### Core Components

- **Fine-tuned Whisper Model**: Whisper-small model fine-tuned on Vietnamese speech data
- **Audio Preprocessing**: Noise reduction, volume normalization, and resampling
- **Real-time Interface**: Gradio web app for live speech recognition
- **Comprehensive Evaluation**: WER, CER, and loss metrics

### Audio Preprocessing Features

- **Noise Reduction**: Background noise removal using noisereduce
- **Volume Normalization**: Standardized audio volume levels
- **Resampling**: Automatic resampling to 16kHz for Whisper compatibility
- **Format Handling**: Support for various audio formats

### Web Interface Features

- **Real-time Transcription**: Live speech-to-text conversion
- **Microphone Input**: Direct recording from browser microphone
- **Vietnamese Optimization**: Specialized for Vietnamese language
- **User-friendly UI**: Clean and intuitive Gradio interface

## 📊 Model Performance

The fine-tuned model shows significant improvements over the base Whisper model for Vietnamese speech:

- **Evaluation Metrics**: Available in `evaluation_results.txt`
- **Word Error Rate (WER)**: [Check evaluation_results.txt]
- **Character Error Rate (CER)**: [Check evaluation_results.txt]
- **Model Loss**: [Check evaluation_results.txt]

## 🔬 Training Process

### Data Preprocessing

1. **Audio Cleaning**: Remove silence, normalize volume
2. **Format Standardization**: Convert to 16kHz mono WAV
3. **Noise Reduction**: Apply noise reduction algorithms
4. **Dataset Splitting**: Train/validation/test split

### Fine-tuning Setup

- **Base Model**: OpenAI Whisper-small
- **Language**: Vietnamese (vi)
- **Task**: Transcription
- **Optimization**: AdamW optimizer with learning rate scheduling
- **Evaluation**: WER and CER metrics

### Training Configuration

```python
# Key training parameters
model_name = "openai/whisper-small"
language = "vi"
task = "transcribe"
batch_size = 16
learning_rate = 1e-5
num_epochs = 3
```

## 📋 Usage Examples

### Using the Web Interface

1. Launch the app: `python app.py`
2. Open the provided URL in your browser
3. Click "Record" and speak in Vietnamese
4. Get real-time transcription results

### Using the Model Programmatically

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa

# Load model
model_path = "./whisper-vi-finetuned/checkpoint-7000"
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)

# Load and preprocess audio
audio, sr = librosa.load("audio_file.wav", sr=16000)
input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features

# Generate transcription
with torch.no_grad():
    generated_ids = model.generate(inputs=input_features)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(transcription)
```

## 🛠️ Development

### Data Preparation

1. Prepare your Vietnamese audio dataset in the required CSV format:
   ```csv
   path,transcription
   audio1.wav,transcription text 1
   audio2.wav,transcription text 2
   ```

2. Run preprocessing:
   ```bash
   python data_preprocess.py
   ```

### Training a New Model

1. Prepare your dataset (CSV files with audio paths and transcriptions)
2. Run the training script:
   ```bash
   python finetune_whisper_vietnamese.py
   ```

3. Monitor training progress and evaluate results

### Evaluation

Run comprehensive evaluation:
```bash
python evaluate_model.py
```

This will generate:
- Word Error Rate (WER)
- Character Error Rate (CER)
- Model loss on test set
- Detailed evaluation report

## 📝 Requirements

### System Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- 10GB+ storage for model and data

### Python Dependencies

```
torch
transformers
datasets
accelerate
jiwer
pandas
numpy
soundfile
librosa
noisereduce
gradio
psutil
scikit-learn
```

## 🔍 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in training configuration
2. **Audio format issues**: Ensure audio files are in supported formats (WAV, MP3, etc.)
3. **Model loading errors**: Check model path and ensure checkpoint exists
4. **Gradio interface issues**: Verify microphone permissions in browser

### Performance Optimization

- Use GPU for faster inference
- Adjust batch size based on available memory
- Consider model quantization for deployment
- Use appropriate audio preprocessing settings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project uses the Whisper model from OpenAI. Please refer to the original Whisper license for usage terms.

## 🙏 Acknowledgments

- OpenAI for the Whisper model
- Hugging Face for the Transformers library
- Gradio team for the web interface framework
- Vietnamese speech data contributors

## 📞 Support

For issues, questions, or contributions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the evaluation results for model performance metrics

---

**Note**: This is a research and educational project. For production use, consider additional testing and validation on your specific use cases. 