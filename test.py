#!/usr/bin/env python3
"""
Multi-Accent Synthetic Stammering Data Generator
Generates synthetic stammering data using various TTS models with different accents
Specifically designed for training stammering detection models with diverse accents
"""

import os
import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
import random
from pathlib import Path
from transformers import pipeline, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

class MultiAccentStammeringGenerator:
    def __init__(self, output_dir="stammering_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "train" / "stammering").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train" / "non_stammering").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "test" / "stammering").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "test" / "non_stammering").mkdir(parents=True, exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_rate = 16000
        
        # Initialize model containers
        self.models = {}
        self.datasets_info = []
        
        # Stammering patterns for different types
        self.stammering_patterns = {
            'repetition': [
                'I-I-I want to go',
                'Th-th-this is difficult',
                'C-c-can you help me',
                'W-w-what time is it',
                'H-h-how are you doing',
                'P-p-please wait for me'
            ],
            'prolongation': [
                'Thiiiiis is a test',
                'Pleeeeease help me',
                'Whaaaat are you doing',
                'Hoooow much does it cost',
                'Wheeeeere is the station',
                'Caaaaan you repeat that'
            ],
            'block': [
                '...I want to speak',
                '...This is hard',
                '...Can you wait',
                '...What should I do',
                '...How do I get there',
                '...Please understand me'
            ]
        }
        
        # Non-stammering sentences for comparison
        self.normal_sentences = [
            'I want to go to the store today.',
            'This is a beautiful day outside.',
            'Can you help me with this problem?',
            'What time does the meeting start?',
            'How are you feeling today?',
            'Please wait for me at the station.',
            'The weather is very nice today.',
            'I need to finish my work soon.',
            'Thank you for your help and support.',
            'Where is the nearest coffee shop?'
        ]
        
        # Accent configurations for different models
        self.accent_configs = {
            'parler_multilingual': [
                ('indian_male', 'A middle-aged Indian man speaking with a clear Hindi accent'),
                ('indian_female', 'A young Indian woman with a pleasant Mumbai accent'),
                ('british_male', 'A British man from London with a clear accent'),
                ('british_female', 'A young British woman with a refined accent'),
                ('american_male', 'An American man from New York with a clear voice'),
                ('american_female', 'A young American woman from California'),
                ('australian_male', 'An Australian man with a distinctive Sydney accent'),
                ('chinese_male', 'A Chinese man speaking English with a Beijing accent'),
                ('japanese_female', 'A Japanese woman speaking English with a Tokyo accent'),
                ('arabic_male', 'An Arabic man speaking English with a Middle Eastern accent'),
                ('south_african_female', 'A South African woman with a Cape Town accent'),
                ('scottish_male', 'A Scottish man from Edinburgh with a strong accent')
            ],
            'mms_languages': [
                ('eng', 'English (General)'),
                ('hin', 'Hindi (India)'),
                ('ben', 'Bengali (India/Bangladesh)'),
                ('ara', 'Arabic'),
                ('cmn', 'Chinese (Mandarin)'),
                ('jpn', 'Japanese'),
                ('fra', 'French'),
                ('deu', 'German'),
                ('spa', 'Spanish'),
                ('rus', 'Russian'),
                ('por', 'Portuguese'),
                ('ita', 'Italian')
            ],
            'indic_parler': [
                ('hindi_male', 'A Hindi speaking man from Delhi'),
                ('hindi_female', 'A young Hindi speaking woman from Mumbai'),
                ('bengali_male', 'A Bengali man from Kolkata'),
                ('tamil_male', 'A Tamil speaking man from Chennai'),
                ('telugu_female', 'A Telugu speaking woman from Hyderabad'),
                ('gujarati_male', 'A Gujarati man from Ahmedabad'),
                ('punjabi_female', 'A Punjabi woman from Amritsar'),
                ('marathi_male', 'A Marathi speaking man from Pune')
            ]
        }
    
    def setup_parler_multilingual(self):
        """Setup Parler-TTS Multilingual model"""
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            
            model_name = "parler-tts/parler-tts-mini-multilingual"
            self.models['parler_multilingual'] = {
                'model': ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(self.device),
                'tokenizer': AutoTokenizer.from_pretrained(model_name)
            }
            print("✅ Parler-TTS Multilingual model loaded")
            return True
        except Exception as e:
            print(f"❌ Failed to load Parler Multilingual: {e}")
            return False
    
    def setup_mms_models(self):
        """Setup Facebook MMS-TTS models for different languages"""
        try:
            self.models['mms'] = {}
            
            # Load a few key language models
            key_languages = ['eng', 'hin', 'ara', 'cmn', 'jpn']
            
            for lang_code in key_languages:
                try:
                    model_name = f"facebook/mms-tts-{lang_code}"
                    tts_pipeline = pipeline("text-to-speech", model=model_name, device=0 if torch.cuda.is_available() else -1)
                    self.models['mms'][lang_code] = tts_pipeline
                    print(f"✅ Loaded MMS-TTS for {lang_code}")
                except Exception as e:
                    print(f"⚠️  Failed to load MMS-TTS for {lang_code}: {e}")
            
            return len(self.models['mms']) > 0
        except Exception as e:
            print(f"❌ Failed to setup MMS models: {e}")
            return False
    
    def setup_indic_parler(self):
        """Setup Indic Parler-TTS model"""
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            
            model_name = "ai4bharat/indic-parler-tts"
            self.models['indic_parler'] = {
                'model': ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(self.device),
                'tokenizer': AutoTokenizer.from_pretrained(model_name)
            }
            print("✅ Indic Parler-TTS model loaded")
            return True
        except Exception as e:
            print(f"⚠️  Indic Parler-TTS not available: {e}")
            return False
    
    def generate_audio_parler(self, text, description, model_key='parler_multilingual'):
        """Generate audio using Parler-TTS models"""
        try:
            model_info = self.models[model_key]
            model = model_info['model']
            tokenizer = model_info['tokenizer']
            
            input_ids = tokenizer(description, return_tensors="pt").input_ids.to(self.device)
            prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(self.device)
            
            generation = model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
                do_sample=True,
                temperature=0.8
            )
            
            audio_arr = generation.cpu().numpy().squeeze()
            return audio_arr, model.config.sampling_rate
        except Exception as e:
            print(f"Error generating audio with Parler: {e}")
            return None, None
    
    def generate_audio_mms(self, text, lang_code):
        """Generate audio using Facebook MMS-TTS"""
        try:
            if lang_code in self.models['mms']:
                tts_pipeline = self.models['mms'][lang_code]
                result = tts_pipeline(text)
                
                if hasattr(result, 'audio'):
                    audio_data = result.audio.flatten()
                    sample_rate = result.sampling_rate
                else:
                    audio_data = result['audio'].flatten()
                    sample_rate = result['sampling_rate']
                
                return audio_data, sample_rate
        except Exception as e:
            print(f"Error generating audio with MMS {lang_code}: {e}")
            return None, None
    
    def add_audio_variations(self, audio, sample_rate):
        """Add variations to make synthetic data more realistic"""
        variations = []
        
        # Original
        variations.append(('original', audio))
        
        # Speed variation
        if random.random() < 0.5:
            speed_factor = random.uniform(0.8, 1.2)
            fast_audio = librosa.effects.time_stretch(audio, rate=speed_factor)
            variations.append(('speed_varied', fast_audio))
        
        # Pitch variation
        if random.random() < 0.5:
            n_steps = random.uniform(-2, 2)
            pitch_audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)
            variations.append(('pitch_varied', pitch_audio))
        
        # Add slight noise
        if random.random() < 0.3:
            noise_factor = 0.005
            noise = np.random.normal(0, noise_factor, audio.shape)
            noisy_audio = audio + noise
            variations.append(('noise_added', noisy_audio))
        
        return variations
    
    def generate_stammering_dataset(self, num_samples_per_accent=20):
        """Generate complete stammering dataset with multiple accents"""
        print("🚀 Starting Multi-Accent Stammering Dataset Generation")
        print("=" * 60)
        
        dataset_entries = []
        file_counter = 0
        
        # Setup models
        models_available = []
        if self.setup_parler_multilingual():
            models_available.append('parler_multilingual')
        if self.setup_mms_models():
            models_available.append('mms')
        if self.setup_indic_parler():
            models_available.append('indic_parler')
        
        if not models_available:
            print("❌ No TTS models available. Please install required packages.")
            return
        
        print(f"✅ Available models: {models_available}")
        
        # Generate stammering data
        print(f"\n🎯 Generating stammering samples...")
        for model_type in models_available:
            if model_type == 'parler_multilingual':
                configs = self.accent_configs['parler_multilingual']
            elif model_type == 'mms':
                configs = self.accent_configs['mms_languages']
            else:  # indic_parler
                configs = self.accent_configs['indic_parler']
            
            for accent_name, accent_desc in configs:
                print(f"  Processing {accent_name}...")
                
                samples_generated = 0
                for pattern_type, patterns in self.stammering_patterns.items():
                    for pattern in patterns[:num_samples_per_accent//6]:  # Distribute across pattern types
                        
                        # Generate audio
                        if model_type in ['parler_multilingual', 'indic_parler']:
                            audio, sr = self.generate_audio_parler(pattern, accent_desc, model_type)
                        else:  # mms
                            audio, sr = self.generate_audio_mms(pattern, accent_name)
                        
                        if audio is not None:
                            # Add variations
                            variations = self.add_audio_variations(audio, sr)
                            
                            for var_name, var_audio in variations:
                                # Determine train/test split (80/20)
                                split = "train" if random.random() < 0.8 else "test"
                                
                                # Save file
                                filename = f"stammering_{accent_name}_{pattern_type}_{file_counter:04d}_{var_name}.wav"
                                filepath = self.output_dir / split / "stammering" / filename
                                
                                # Resample to target sample rate
                                if sr != self.sample_rate:
                                    var_audio = librosa.resample(var_audio, orig_sr=sr, target_sr=self.sample_rate)
                                
                                sf.write(filepath, var_audio, self.sample_rate)
                                
                                # Add to dataset
                                dataset_entries.append({
                                    'filename': str(filepath),
                                    'label': 'stammering',
                                    'split': split,
                                    'accent': accent_name,
                                    'model': model_type,
                                    'pattern_type': pattern_type,
                                    'variation': var_name,
                                    'text': pattern
                                })
                                
                                file_counter += 1
                                samples_generated += 1
                
                print(f"    Generated {samples_generated} samples for {accent_name}")
        
        # Generate non-stammering data
        print(f"\n🎯 Generating non-stammering samples...")
        for model_type in models_available:
            if model_type == 'parler_multilingual':
                configs = self.accent_configs['parler_multilingual']
            elif model_type == 'mms':
                configs = self.accent_configs['mms_languages']
            else:
                configs = self.accent_configs['indic_parler']
            
            for accent_name, accent_desc in configs:
                print(f"  Processing {accent_name}...")
                
                samples_generated = 0
                for sentence in self.normal_sentences[:num_samples_per_accent//2]:
                    
                    # Generate audio
                    if model_type in ['parler_multilingual', 'indic_parler']:
                        audio, sr = self.generate_audio_parler(sentence, accent_desc, model_type)
                    else:
                        audio, sr = self.generate_audio_mms(sentence, accent_name)
                    
                    if audio is not None:
                        # Add variations (fewer for non-stammering)
                        variations = self.add_audio_variations(audio, sr)
                        
                        for var_name, var_audio in variations[:2]:  # Limit variations
                            split = "train" if random.random() < 0.8 else "test"
                            
                            filename = f"normal_{accent_name}_{file_counter:04d}_{var_name}.wav"
                            filepath = self.output_dir / split / "non_stammering" / filename
                            
                            if sr != self.sample_rate:
                                var_audio = librosa.resample(var_audio, orig_sr=sr, target_sr=self.sample_rate)
                            
                            sf.write(filepath, var_audio, self.sample_rate)
                            
                            dataset_entries.append({
                                'filename': str(filepath),
                                'label': 'non_stammering',
                                'split': split,
                                'accent': accent_name,
                                'model': model_type,
                                'pattern_type': 'normal',
                                'variation': var_name,
                                'text': sentence
                            })
                            
                            file_counter += 1
                            samples_generated += 1
                
                print(f"    Generated {samples_generated} normal samples for {accent_name}")
        
        # Save dataset metadata
        df = pd.DataFrame(dataset_entries)
        df.to_csv(self.output_dir / 'dataset_metadata.csv', index=False)
        
        # Create labels.csv for Wav2Vec2 training (from your roadmap)
        labels_df = df[['filename', 'label', 'split']].copy()
        labels_df.to_csv(self.output_dir / 'labels.csv', index=False)
        
        self.print_dataset_summary(df)
    
    def print_dataset_summary(self, df):
        """Print comprehensive dataset summary"""
        print("\n" + "="*60)
        print("📊 DATASET GENERATION SUMMARY")
        print("="*60)
        
        total_files = len(df)
        train_files = len(df[df['split'] == 'train'])
        test_files = len(df[df['split'] == 'test'])
        
        print(f"📁 Total files generated: {total_files}")
        print(f"   📚 Training files: {train_files}")
        print(f"   🧪 Test files: {test_files}")
        
        print(f"\n🎭 By Label:")
        for label in df['label'].unique():
            count = len(df[df['label'] == label])
            print(f"   {label}: {count} files")
        
        print(f"\n🌍 By Accent:")
        accent_counts = df['accent'].value_counts()
        for accent, count in accent_counts.items():
            print(f"   {accent}: {count} files")
        
        print(f"\n🤖 By Model:")
        model_counts = df['model'].value_counts()
        for model, count in model_counts.items():
            print(f"   {model}: {count} files")
        
        print(f"\n📋 Files saved in: {self.output_dir.absolute()}")
        print(f"   • Audio files organized in train/test folders")
        print(f"   • Metadata: dataset_metadata.csv")
        print(f"   • Labels for training: labels.csv")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Review generated audio files")
        print(f"   2. Run your Wav2Vec2 training script")
        print(f"   3. Adjust num_samples_per_accent if needed")
        print(f"   4. Add real stammering data if available")

def main():
    """Main function to run the multi-accent stammering generator"""
    print("🌍 Multi-Accent Synthetic Stammering Data Generator")
    print("=" * 60)
    print("This script generates synthetic stammering data with various accents:")
    print("  • Indian accents (Hindi, Bengali, Tamil, etc.)")
    print("  • International accents (British, American, Australian, etc.)")
    print("  • Asian accents (Chinese, Japanese)")
    print("  • Middle Eastern accents (Arabic)")
    print("=" * 60)
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    required_packages = [
        "torch", "torchaudio", "transformers", "librosa", 
        "soundfile", "numpy", "pandas"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\nAdditional TTS packages:")
        print("pip install parler-tts")
    
    # Run generator
    generator = MultiAccentStammeringGenerator()
    
    # Ask user for number of samples
    try:
        num_samples = int(input(f"\n🎯 How many samples per accent? (recommended: 15-25): ") or "20")
    except:
        num_samples = 20
    
    print(f"\n🚀 Starting generation with {num_samples} samples per accent...")
    generator.generate_stammering_dataset(num_samples_per_accent=num_samples)
    
    print(f"\n✨ Generation complete! Dataset ready for Wav2Vec2 training.")

if __name__ == "__main__":
    main()