#!/usr/bin/env python
# coding: utf-8

"""
Model Evaluation Script for Vietnamese Whisper Fine-tuned Model
Evaluates the model on test set with Loss, WER, and CER metrics
"""

import os
import gc
import torch
import pandas as pd
import numpy as np
from datasets import Dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import jiwer
from tqdm import tqdm

def print_memory_usage():
    """Print current memory usage"""
    import psutil
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

def load_test_dataset(csv_path):
    """Load and prepare test dataset"""
    print(f"Loading test dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Test dataset size: {len(df)} samples")
    
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

def calculate_loss(model, dataloader, device):
    """Calculate average loss on the dataset"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    print("Calculating loss...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing loss"):
            # Move batch to device
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss
            
            total_loss += loss.item()
            num_batches += 1
            
            # Clean up GPU memory
            del input_features, labels, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss

def generate_predictions(model, processor, dataloader, device):
    """Generate predictions for WER and CER calculation"""
    model.eval()
    predictions = []
    references = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            # Move input to device
            input_features = batch["input_features"].to(device)
            labels = batch["labels"]
            
            # Generate predictions
            generated_ids = model.generate(
                input_features,
                max_length=448,
                num_beams=1,
                do_sample=False,
                task="transcribe",
                language="vi"
            )
            
            # Decode predictions
            pred_str = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Decode references (replace -100 with pad token id)
            labels[labels == -100] = processor.tokenizer.pad_token_id
            label_str = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            predictions.extend(pred_str)
            references.extend(label_str)
            
            # Clean up GPU memory
            del input_features, generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return predictions, references

def calculate_wer(predictions, references):
    """Calculate Word Error Rate"""
    try:
        # Clean and normalize text
        clean_predictions = []
        clean_references = []
        
        for pred, ref in zip(predictions, references):
            # Remove extra whitespace and normalize
            pred_clean = " ".join(pred.strip().split())
            ref_clean = " ".join(ref.strip().split())
            
            clean_predictions.append(pred_clean)
            clean_references.append(ref_clean)
        
        wer_score = jiwer.wer(clean_references, clean_predictions)
        return wer_score
    except Exception as e:
        print(f"Error calculating WER: {e}")
        return 1.0

def calculate_cer(predictions, references):
    """Calculate Character Error Rate"""
    try:
        # Clean text but keep character level
        clean_predictions = []
        clean_references = []
        
        for pred, ref in zip(predictions, references):
            # Remove extra whitespace but keep all characters
            pred_clean = pred.strip()
            ref_clean = ref.strip()
            
            clean_predictions.append(pred_clean)
            clean_references.append(ref_clean)
        
        cer_score = jiwer.cer(clean_references, clean_predictions)
        return cer_score
    except Exception as e:
        print(f"Error calculating CER: {e}")
        return 1.0

def main():
    """Main evaluation function"""
    print("=== Vietnamese Whisper Model Evaluation ===")
    print_memory_usage()
    
    # Configuration
    model_path = "./whisper-vi-finetuned/checkpoint-7000"
    test_csv = "fpt_test.csv"
    batch_size = 2  # Small batch size for memory efficiency
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    if not os.path.exists(test_csv):
        print(f"Error: Test dataset not found at {test_csv}")
        return
    
    # Load model and processor
    print("Loading model and processor...")
    try:
        processor = WhisperProcessor.from_pretrained(model_path)
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
        model.to(device)
        print("Model loaded successfully!")
        print_memory_usage()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load test dataset
    try:
        test_dataset = load_test_dataset(test_csv)
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return
    
    # Preprocess test dataset
    print("Preprocessing test dataset...")
    try:
        test_dataset = test_dataset.map(
            lambda batch: prepare_dataset(batch, processor),
            batched=True,
            batch_size=4,
            num_proc=1,
            remove_columns=test_dataset.column_names,
            load_from_cache_file=True,
            keep_in_memory=False,
            desc="Processing test data"
        )
        
        gc.collect()
        print("Test dataset preprocessed successfully!")
        print_memory_usage()
    except Exception as e:
        print(f"Error preprocessing dataset: {e}")
        return
    
    # Create data collator and dataloader
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # Create dataloader
    from torch.utils.data import DataLoader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False,
        pin_memory=False,
        num_workers=0
    )
    
    print(f"Created dataloader with {len(test_dataloader)} batches")
    
    # Evaluation metrics
    results = {}
    
    # 1. Calculate Loss
    print("\n" + "="*50)
    print("1. CALCULATING LOSS")
    print("="*50)
    try:
        avg_loss = calculate_loss(model, test_dataloader, device)
        results["loss"] = avg_loss
        print(f"Average Loss: {avg_loss:.4f}")
    except Exception as e:
        print(f"Error calculating loss: {e}")
        results["loss"] = None
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_memory_usage()
    
    # 2. Generate predictions for WER and CER
    print("\n" + "="*50)
    print("2. GENERATING PREDICTIONS")
    print("="*50)
    try:
        predictions, references = generate_predictions(model, processor, test_dataloader, device)
        print(f"Generated {len(predictions)} predictions")
        
        # Show a few examples
        print("\nExample predictions:")
        for i in range(min(3, len(predictions))):
            print(f"Reference: {references[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error generating predictions: {e}")
        predictions, references = None, None
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_memory_usage()
    
    if predictions and references:
        # 3. Calculate WER
        print("\n" + "="*50)
        print("3. CALCULATING WER (Word Error Rate)")
        print("="*50)
        try:
            wer_score = calculate_wer(predictions, references)
            results["wer"] = wer_score
            print(f"WER: {wer_score:.4f} ({wer_score*100:.2f}%)")
        except Exception as e:
            print(f"Error calculating WER: {e}")
            results["wer"] = None
        
        # 4. Calculate CER
        print("\n" + "="*50)
        print("4. CALCULATING CER (Character Error Rate)")
        print("="*50)
        try:
            cer_score = calculate_cer(predictions, references)
            results["cer"] = cer_score
            print(f"CER: {cer_score:.4f} ({cer_score*100:.2f}%)")
        except Exception as e:
            print(f"Error calculating CER: {e}")
            results["cer"] = None
    
    # Final Results
    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Test samples: {len(test_dataset)}")
    print("-" * 60)
    
    if results.get("loss") is not None:
        print(f"Loss:     {results['loss']:.4f}")
    else:
        print("Loss:     Failed to calculate")
    
    if results.get("wer") is not None:
        print(f"WER:      {results['wer']:.4f} ({results['wer']*100:.2f}%)")
    else:
        print("WER:      Failed to calculate")
    
    if results.get("cer") is not None:
        print(f"CER:      {results['cer']:.4f} ({results['cer']*100:.2f}%)")
    else:
        print("CER:      Failed to calculate")
    
    print("="*60)
    
    # Save results to file
    results_file = "evaluation_results.txt"
    try:
        with open(results_file, "w", encoding="utf-8") as f:
            f.write("Vietnamese Whisper Model Evaluation Results\n")
            f.write("="*50 + "\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Test samples: {len(test_dataset)}\n")
            f.write(f"Loss: {results.get('loss', 'N/A')}\n")
            f.write(f"WER: {results.get('wer', 'N/A')}\n")
            f.write(f"CER: {results.get('cer', 'N/A')}\n")
            
            if predictions and references:
                f.write("\nSample Predictions:\n")
                f.write("-" * 30 + "\n")
                for i in range(min(10, len(predictions))):
                    f.write(f"Sample {i+1}:\n")
                    f.write(f"Reference:  {references[i]}\n")
                    f.write(f"Prediction: {predictions[i]}\n")
                    f.write("-" * 30 + "\n")
        
        print(f"Results saved to: {results_file}")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    print("\nEvaluation completed!")
    print_memory_usage()

if __name__ == "__main__":
    main() 