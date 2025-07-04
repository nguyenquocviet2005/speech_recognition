#!/usr/bin/env python3

"""
Fine-tuning Whisper for Vietnamese ASR

This script demonstrates how to fine-tune OpenAI's Whisper model on a Vietnamese speech dataset 
using Hugging Face Transformers with memory-efficient settings.
"""

import os
import gc
import psutil
import pandas as pd
from datasets import Dataset, Audio
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

def load_csv_to_dataset(csv_path):
    """Load CSV file and convert to HuggingFace Dataset with Audio column"""
    df = pd.read_csv(csv_path)
    ds = Dataset.from_pandas(df)
    ds = ds.cast_column("path", Audio(sampling_rate=16000))
    return ds

def prepare_dataset(batch, processor):
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

def main():
    print("Starting Whisper Vietnamese Fine-tuning...")
    print("=" * 50)
    
    # Check initial memory
    print("Initial system state:")
    print_memory_usage()
    print()
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = load_csv_to_dataset("fpt_train.csv")
    val_dataset = load_csv_to_dataset("fpt_val.csv")
    test_dataset = load_csv_to_dataset("fpt_test.csv")
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print("Datasets loaded.")
    print_memory_usage()
    print()
    
    # Load Whisper model and processor
    print("Loading Whisper model and processor...")
    model_name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
    
    print("Model loaded.")
    print_memory_usage()
    print()
    
    # Preprocessing with memory management
    print("Starting data preprocessing...")
    print("This may take several minutes for large datasets...")
    
    # Create a partial function with processor
    def prepare_batch(batch):
        return prepare_dataset(batch, processor)
    
    # Process training dataset
    print("Processing training data...")
    train_dataset = train_dataset.map(
        prepare_batch,
        batched=True,
        batch_size=4,  # Small batch size to prevent memory overflow
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
    print()
    
    # Process validation dataset
    print("Processing validation data...")
    val_dataset = val_dataset.map(
        prepare_batch,
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
    print()
    
    # Create data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Setup training arguments
    print("Setting up training configuration...")
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
    
    # Define metrics computation
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
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    
    print("Trainer initialized.")
    print_memory_usage()
    print()
    
    # Start training
    print("Starting training...")
    print("=" * 50)
    trainer.train()
    
    print("Training completed!")
    print_memory_usage()
    print()
    
    # Evaluate on test set
    print("Processing test data for evaluation...")
    test_dataset = test_dataset.map(
        prepare_batch,
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
    
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print("Test Results:")
    print(test_results)
    
    # Save the fine-tuned model
    print("Saving fine-tuned model...")
    model.save_pretrained("./whisper-vi-finetuned")
    processor.save_pretrained("./whisper-vi-finetuned")
    print("Model and processor saved to ./whisper-vi-finetuned")
    
    print("=" * 50)
    print("Fine-tuning completed successfully!")
    print_memory_usage()

if __name__ == "__main__":
    main() 