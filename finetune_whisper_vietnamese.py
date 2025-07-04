#!/usr/bin/env python
# coding: utf-8

# # Fine-tuning Whisper for Vietnamese ASR
# This notebook demonstrates how to fine-tune OpenAI's Whisper model on a Vietnamese speech dataset using Hugging Face Transformers. The workflow includes environment setup, data loading, preprocessing, model training, and evaluation.

# ## 1. Environment Setup
# Install the required libraries: `transformers`, `datasets`, `torchaudio`, and `jiwer` for evaluation.

# In[2]:


# get_ipython().system('pip install transformers datasets torchaudio jiwer --quiet')


# In[3]:


import os
import gc
import psutil
import pandas as pd
from datasets import load_dataset, Dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, TrainingArguments, Trainer
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

def print_memory_usage():
    """Print current memory usage"""
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent:.1f}% ({memory.used / (1024**3):.2f}/{memory.total / (1024**3):.2f} GB)")
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB allocated")

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# ## 2. Load and Prepare Data
# Assume your CSV files (`fpt_train.csv`, `fpt_val.csv`, `fpt_test.csv`) are in the current directory and contain columns: `path` (audio file path) and `transcription` (text).

# In[4]:


def load_csv_to_dataset(csv_path):
    df = pd.read_csv(csv_path)
    ds = Dataset.from_pandas(df)
    ds = ds.cast_column("path", Audio(sampling_rate=16000))
    return ds

print("Loading datasets...")
print_memory_usage()

train_dataset = load_csv_to_dataset("fpt_train.csv")
val_dataset = load_csv_to_dataset("fpt_val.csv")
test_dataset = load_csv_to_dataset("fpt_test.csv")

print("Datasets loaded.")
print_memory_usage()

# ## 3. Load Whisper Model and Processor

# In[5]:


model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")

print("Model loaded.")
print_memory_usage()

# ## 4. Memory-efficient Preprocessing Function
# Process data in smaller chunks and clean up memory aggressively

# In[ ]:


def prepare_dataset(batch):
    """Memory-efficient preprocessing function"""
    audio = batch["path"]
    
    # Process audio - handle both single item and batch
    if isinstance(audio, list):
        # Batched processing
        input_features = []
        for audio_item in audio:
            features = processor.feature_extractor(
                audio_item["array"], 
                sampling_rate=audio_item["sampling_rate"]
            ).input_features[0]
            input_features.append(features)
    else:
        # Single item processing
        input_features = processor.feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
    
    # Process transcripts
    if isinstance(batch["transcription"], list):
        labels = [processor.tokenizer(text).input_ids for text in batch["transcription"]]
    else:
        labels = processor.tokenizer(batch["transcription"]).input_ids
    
    return {"input_features": input_features, "labels": labels}

print("Starting data preprocessing...")
print_memory_usage()

# Process datasets with very small batch sizes and aggressive memory management
train_dataset = train_dataset.map(
    prepare_dataset,
    batched=True,
    batch_size=4,  # Very small batch size to prevent memory overflow
    num_proc=1,
    remove_columns=train_dataset.column_names,  # Remove original columns to save memory
    load_from_cache_file=True,
    keep_in_memory=False,
    desc="Processing training data"
)

# Force garbage collection
gc.collect()
print("Training data processed.")
print_memory_usage()

val_dataset = val_dataset.map(
    prepare_dataset,
    batched=True,
    batch_size=4,
    num_proc=1,
    remove_columns=val_dataset.column_names,
    load_from_cache_file=True,
    keep_in_memory=False,
    desc="Processing validation data"
)

# Force garbage collection
gc.collect()
print("Validation data processed.")
print_memory_usage()

# Create data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# ## 5. Training Arguments and Trainer

# In[ ]:


training_args = TrainingArguments(
    output_dir="./whisper-vi-finetuned",
    per_device_train_batch_size=2,  # Reduced batch size for memory efficiency
    per_device_eval_batch_size=2,   # Also reduce eval batch size
    gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
    eval_strategy="steps",
    num_train_epochs=3,
    save_steps=1000,  # Save less frequently to save disk space
    eval_steps=1000,
    logging_steps=100,
    learning_rate=1e-4,
    warmup_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=False,  # Reduce memory usage
    dataloader_num_workers=0,     # Reduce memory usage
    push_to_hub=False,
    remove_unused_columns=True,   # Remove unused columns
    prediction_loss_only=True,    # Only compute loss during evaluation
)

def compute_metrics(pred):
    try:
        from jiwer import wer
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # decode predictions and labels
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        wer_score = wer(label_str, pred_str)
        # Clean up to free memory
        del pred_str, label_str
        gc.collect()
        return {"wer": wer_score}
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {"wer": 1.0}  # Return worst possible score if error

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,  # Updated from tokenizer to avoid deprecation warning
)

print("Trainer initialized.")
print_memory_usage()

# ## 6. Start Training

# In[ ]:


print("Starting training...")
trainer.train()

# ## 7. Evaluate on Test Set

# In[ ]:


# Process test data for evaluation
test_dataset = test_dataset.map(
    prepare_dataset,
    batched=True,
    batch_size=4,
    num_proc=1,
    remove_columns=test_dataset.column_names,
    load_from_cache_file=True,
    keep_in_memory=False,
    desc="Processing test data"
)

gc.collect()
print("Test data processed.")
print_memory_usage()

test_results = trainer.evaluate(test_dataset)
print(test_results)

# In[ ]:


# Save the fine-tuned model and processor for later use
model.save_pretrained("./whisper-vi-finetuned")
processor.save_pretrained("./whisper-vi-finetuned")
print("Model and processor saved to ./whisper-vi-finetuned")

# ---
# This notebook provides a memory-efficient pipeline for fine-tuning Whisper on Vietnamese ASR data.
