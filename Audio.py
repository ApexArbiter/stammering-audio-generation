import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import json
import random
from typing import List, Dict, Tuple, Optional
import scipy.signal
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

# ESPnet imports
try:
    from espnet2.bin.tts_inference import Text2Speech
    from espnet_model_zoo.downloader import ModelDownloader
    ESPNET_AVAILABLE = True
    print("✅ ESPnet libraries loaded successfully")
except ImportError:
    ESPNET_AVAILABLE = False
    print("❌ ESPnet not available. Install with: pip install espnet espnet_model_zoo")

class ESPnetCMUArcticSynthesizer:
    """
    ESPnet-based CMU Arctic synthesizer for stammering voice generation with multi-speaker support
    """
    
    def __init__(self, 
                 model_tag: str = "espnet/kan-bayashi_vctk_tts_train_full_band_multi_spk_vits_raw_phn_tacotron_g-truncated-50b003",
                 sample_rate: int = 22050,
                 output_dir: str = "synthetic_stammering_data"):
        
        self.sample_rate = sample_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "raw_synthetic").mkdir(exist_ok=True)
        (self.output_dir / "augmented").mkdir(exist_ok=True)
        (self.output_dir / "processed").mkdir(exist_ok=True)
        
        # Setup ESPnet TTS model
        self.model_tag = model_tag
        self.speaker_id = None  # Will be set during model setup
        self.available_speakers = []
        self.spk2id = {}  # Speaker name to ID mapping
        self.id2spk = {}  # ID to speaker name mapping
        self.setup_espnet_model(model_tag)
        
        # Augmentation parameters
        self.augmentation_params = {
            'noise_levels': [0.005, 0.01, 0.02, 0.03],
            'pitch_shifts': [-2, -1, 1, 2],  # semitones
            'speed_factors': [0.85, 0.9, 1.1, 1.15],
            'volume_factors': [0.8, 0.9, 1.1, 1.2]
        }
    
    def setup_espnet_model(self, model_tag: str):
        """Setup ESPnet TTS model with proper speaker handling"""
        if not ESPNET_AVAILABLE:
            print("❌ Cannot setup ESPnet model - library not installed")
            self.tts_model = None
            return
            
        try:
            print(f"🔄 Loading ESPnet model: {model_tag}")
            print("This may take a few minutes for first-time download...")
            
            # Initialize the TTS model
            self.tts_model = Text2Speech.from_pretrained(model_tag)
            
            # Get available speakers if it's a multi-speaker model
            self.check_speaker_info()
            
            print(f"✅ ESPnet TTS model loaded successfully: {model_tag}")
            if self.available_speakers:
                print(f"📢 Available speakers: {len(self.available_speakers)}")
                print(f"🎤 Default speaker ID: {self.speaker_id}")
                # Print some example speakers
                example_speakers = list(self.spk2id.items())[:10]
                print(f"📝 Example speakers: {example_speakers}")
            
        except Exception as e:
            print(f"❌ Error loading ESPnet model: {e}")
            print("Falling back to simpler model...")
            self.fallback_to_single_speaker()
    
    def fallback_to_single_speaker(self):
        """Fallback to single speaker models if multi-speaker fails"""
        try:
            # Try a simpler single-speaker model
            simple_models = [
                "espnet/kan-bayashi_ljspeech_vits",
                "espnet/kan-bayashi_ljspeech_tts_train_tacotron2_raw_phn_tacotron_g-truncated-50b003",
                "espnet/kan-bayashi_ljspeech_tts_train_fastspeech_raw_phn_tacotron_g-truncated-50b003"
            ]
            
            for simple_model in simple_models:
                try:
                    print(f"🔄 Trying simpler model: {simple_model}")
                    self.tts_model = Text2Speech.from_pretrained(simple_model)
                    self.model_tag = simple_model
                    self.check_speaker_info()
                    print(f"✅ Successfully loaded: {simple_model}")
                    break
                except Exception as e2:
                    print(f"❌ Failed to load {simple_model}: {e2}")
                    continue
            else:
                raise Exception("All model loading attempts failed")
                
        except Exception as e3:
            print(f"❌ All model loading attempts failed: {e3}")
            self.tts_model = None
    
    def check_speaker_info(self):
        """Check if the model supports multiple speakers and get speaker info"""
        try:
            # Check for speaker-related attributes in the model
            if hasattr(self.tts_model, 'spk2id') and self.tts_model.spk2id is not None:
                self.spk2id = self.tts_model.spk2id
                self.id2spk = {v: k for k, v in self.spk2id.items()}
                self.available_speakers = list(self.spk2id.keys())
                # Use first available speaker as default
                self.speaker_id = 0
                print(f"🔍 Multi-speaker model detected with {len(self.available_speakers)} speakers")
                print(f"🎭 Speaker mapping loaded: {len(self.spk2id)} speakers")
                
            elif hasattr(self.tts_model.tts.generator, 'spk2id') and self.tts_model.tts.generator.spk2id is not None:
                # Sometimes the mapping is in the generator
                self.spk2id = self.tts_model.tts.generator.spk2id
                self.id2spk = {v: k for k, v in self.spk2id.items()}
                self.available_speakers = list(self.spk2id.keys())
                self.speaker_id = 0
                print(f"🔍 Multi-speaker model detected (via generator) with {len(self.available_speakers)} speakers")
                
            else:
                # Try to detect from model config or other attributes
                if hasattr(self.tts_model, 'tts') and hasattr(self.tts_model.tts, 'spk_embed_dim'):
                    # Model supports speakers but mapping not available
                    # Create dummy mapping for VCTK (common case)
                    if 'vctk' in self.model_tag.lower():
                        # VCTK has speakers p225, p226, ..., p376 (but not all are present)
                        # Create a reasonable range
                        vctk_speakers = [f"p{225 + i}" for i in range(50)]  # First 50 speakers
                        self.spk2id = {spk: i for i, spk in enumerate(vctk_speakers)}
                        self.id2spk = {i: spk for spk, i in self.spk2id.items()}
                        self.available_speakers = vctk_speakers
                        self.speaker_id = 0
                        print(f"🔍 VCTK multi-speaker model detected, created speaker mapping for {len(vctk_speakers)} speakers")
                    else:
                        # Generic multi-speaker model
                        num_speakers = min(getattr(self.tts_model.tts, 'spk_embed_dim', 100), 100)
                        generic_speakers = [f"speaker_{i:03d}" for i in range(num_speakers)]
                        self.spk2id = {spk: i for i, spk in enumerate(generic_speakers)}
                        self.id2spk = {i: spk for spk, i in self.spk2id.items()}
                        self.available_speakers = generic_speakers
                        self.speaker_id = 0
                        print(f"🔍 Generic multi-speaker model detected, created mapping for {num_speakers} speakers")
                else:
                    # Single speaker model
                    self.available_speakers = []
                    self.speaker_id = None
                    self.spk2id = {}
                    self.id2spk = {}
                    print("🔍 Single-speaker model detected")
                
        except Exception as e:
            print(f"⚠️  Could not determine speaker info: {e}")
            self.available_speakers = []
            self.speaker_id = None
            self.spk2id = {}
            self.id2spk = {}
    
    def get_random_speaker_id(self) -> Optional[int]:
        """Get a random speaker ID for variation"""
        if self.available_speakers:
            return random.randint(0, len(self.available_speakers) - 1)
        return None
    
    def get_speaker_name(self, speaker_id: int) -> str:
        """Get speaker name from ID"""
        return self.id2spk.get(speaker_id, f"speaker_{speaker_id}")
    
    def synthesize_speech(self, text: str, output_path: str = None, speaker_id: Optional[int] = None) -> np.ndarray:
        """Synthesize speech from text using ESPnet with proper speaker handling"""
        if self.tts_model is None:
            print("❌ TTS model not available, using fallback synthesis")
            return self.fallback_synthesis(text)
        
        try:
            # Clean text for TTS
            cleaned_text = self.clean_text_for_tts(text)
            
            # Determine speaker ID
            if speaker_id is None:
                if self.available_speakers:
                    speaker_id = self.speaker_id  # Use default
                else:
                    speaker_id = None  # Single speaker model
            
            # Validate speaker ID
            if self.available_speakers and speaker_id is not None:
                speaker_id = max(0, min(speaker_id, len(self.available_speakers) - 1))
                speaker_name = self.get_speaker_name(speaker_id)
                print(f"🎵 Synthesizing: '{cleaned_text[:50]}...' with speaker {speaker_name} (ID: {speaker_id})")
            else:
                print(f"🎵 Synthesizing: '{cleaned_text[:50]}...' (single speaker)")
            
            # Prepare synthesis parameters
            synthesis_params = {"text": cleaned_text}
            
            # Add speaker ID if required (this is the key fix!)
            if self.available_speakers and speaker_id is not None:
                # Different models may expect different parameter names
                # Try the most common ones
                if hasattr(self.tts_model, 'spk2id') or 'vits' in self.model_tag.lower():
                    # VITS models typically use 'sids'
                    synthesis_params["sids"] = torch.tensor([speaker_id], dtype=torch.long)
                elif 'fastspeech' in self.model_tag.lower():
                    # FastSpeech models might use 'spembs' or 'sids'
                    synthesis_params["sids"] = torch.tensor([speaker_id], dtype=torch.long)
                else:
                    # Generic approach
                    synthesis_params["sids"] = torch.tensor([speaker_id], dtype=torch.long)
                
                print(f"🎭 Using speaker tensor: {synthesis_params.get('sids', 'None')}")
            
            # Generate speech
            with torch.no_grad():
                output_dict = self.tts_model(**synthesis_params)
            
            speech_audio = output_dict["wav"].detach().cpu().numpy()
            
            # Handle different output formats
            if speech_audio.ndim > 1:
                speech_audio = speech_audio.flatten()
            
            # Ensure correct data type
            if speech_audio.dtype != np.float32:
                speech_audio = speech_audio.astype(np.float32)
            
            # Normalize audio
            if np.max(np.abs(speech_audio)) > 0:
                speech_audio = speech_audio / (np.max(np.abs(speech_audio)) + 1e-7)
            
            # Save if path provided
            if output_path:
                sf.write(output_path, speech_audio, self.sample_rate)
                speaker_info = f" (Speaker: {self.get_speaker_name(speaker_id)})" if speaker_id is not None else ""
                print(f"💾 Saved: {output_path}{speaker_info}")
            
            return speech_audio
            
        except Exception as e:
            print(f"❌ Synthesis error: {e}")
            print("Using fallback synthesis...")
            return self.fallback_synthesis(text)
    
    def clean_text_for_tts(self, text: str) -> str:
        """Clean stammering text for better TTS synthesis"""
        # Remove excessive repetition markers
        cleaned = text.replace('-', ' ')  # Replace dashes with spaces
        
        # Handle prolongations (convert aaaa to a)
        import re
        cleaned = re.sub(r'([aeiou])\1{2,}', r'\1', cleaned)
        
        # Remove blocks (dots)
        cleaned = cleaned.replace('...', ' ')
        cleaned = cleaned.replace('....', ' ')
        cleaned = cleaned.replace('.....', ' ')
        cleaned = cleaned.replace('......', ' ')
        
        # Clean up multiple spaces
        cleaned = ' '.join(cleaned.split())
        
        # Ensure text ends with punctuation for better synthesis
        if not cleaned.endswith(('.', '!', '?')):
            cleaned += '.'
        
        return cleaned
    
    def fallback_synthesis(self, text: str) -> np.ndarray:
        """Fallback synthesis using simple method"""
        print("🔄 Using fallback synthesis (basic audio generation)")
        
        # Create a simple synthetic audio based on text length
        duration = max(len(text.split()) * 0.6, 1.0)  # ~0.6 seconds per word, minimum 1 second
        samples = int(duration * self.sample_rate)
        
        # Generate simple synthetic speech-like audio
        t = np.linspace(0, duration, samples)
        
        # Create formant-like frequencies (simplified speech)
        f1 = 800 + 200 * np.sin(2 * np.pi * 2 * t)  # First formant
        f2 = 1200 + 300 * np.sin(2 * np.pi * 3 * t)  # Second formant
        f3 = 2500 + 400 * np.sin(2 * np.pi * 1.5 * t)  # Third formant
        
        # Generate speech-like waveform
        audio = (0.4 * np.sin(2 * np.pi * f1 * t) + 
                0.3 * np.sin(2 * np.pi * f2 * t) +
                0.1 * np.sin(2 * np.pi * f3 * t))
        
        # Add some harmonic content
        fundamental = 150 + 50 * np.sin(2 * np.pi * 1.2 * t)
        audio += 0.2 * np.sin(2 * np.pi * fundamental * t)
        
        # Add stammering artifacts based on text patterns
        if any(pattern in text for pattern in ['-', ':', '...']):
            # Add repetitive patterns for stammering
            segment_length = len(audio) // 8
            for i in range(0, len(audio) - segment_length, segment_length * 2):
                # Create emphasis for stammering
                audio[i:i+segment_length//2] *= 1.3
                # Add slight pause
                if i + segment_length < len(audio):
                    audio[i+segment_length//2:i+segment_length] *= 0.3
        
        # Apply realistic envelope
        envelope = np.exp(-np.abs(t - duration/2) / (duration/3))
        envelope = np.maximum(envelope, 0.1)  # Minimum envelope level
        audio = audio * envelope
        
        # Add slight noise for realism
        noise = np.random.normal(0, 0.005, len(audio))
        audio = audio + noise
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / (np.max(np.abs(audio)) + 1e-7)
        
        return audio.astype(np.float32)
    
    def synthesize_with_different_speakers(self, text: str, num_variations: int = 3) -> List[Tuple[np.ndarray, int]]:
        """Synthesize with different speakers if available"""
        audios = []
        
        if not self.available_speakers:
            # Single speaker model - just generate multiple times with slight variations
            for i in range(num_variations):
                audio = self.synthesize_speech(text)
                audios.append((audio, None))
        else:
            # Multi-speaker model - use different speakers
            num_speakers_to_use = min(num_variations, len(self.available_speakers), 10)
            speaker_ids = random.sample(range(len(self.available_speakers)), num_speakers_to_use)
            
            for spk_id in speaker_ids:
                audio = self.synthesize_speech(text, speaker_id=spk_id)
                audios.append((audio, spk_id))
        
        return audios
    
    def add_noise(self, audio: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add realistic background noise"""
        noise = np.random.normal(0, noise_level, len(audio))
        return audio + noise
    
    def pitch_shift(self, audio: np.ndarray, n_steps: float) -> np.ndarray:
        """Shift pitch by n semitones"""
        try:
            return librosa.effects.pitch_shift(
                audio, 
                sr=self.sample_rate, 
                n_steps=n_steps
            )
        except Exception as e:
            print(f"⚠️  Pitch shift error: {e}")
            return audio
    
    def time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """Change speed without affecting pitch"""
        try:
            return librosa.effects.time_stretch(audio, rate=rate)
        except Exception as e:
            print(f"⚠️  Time stretch error: {e}")
            return audio
    
    def volume_change(self, audio: np.ndarray, factor: float) -> np.ndarray:
        """Adjust volume"""
        return np.clip(audio * factor, -1.0, 1.0)
    
    def add_reverb(self, audio: np.ndarray, room_size: float = 0.2) -> np.ndarray:
        """Add simple reverb effect"""
        try:
            # Create impulse response
            impulse_length = int(0.3 * self.sample_rate)
            impulse = np.exp(-np.arange(impulse_length) / (room_size * impulse_length))
            impulse = np.random.randn(impulse_length) * impulse * 0.1
            
            # Convolve with audio
            reverb_audio = np.convolve(audio, impulse, mode='same')
            return 0.8 * audio + 0.2 * reverb_audio
        except Exception as e:
            print(f"⚠️  Reverb error: {e}")
            return audio
    
    def augment_audio(self, audio: np.ndarray, num_augmentations: int = 3) -> List[np.ndarray]:
        """Generate multiple augmented versions"""
        augmented_audios = []
        
        for i in range(num_augmentations):
            augmented = audio.copy()
            
            # Apply random augmentations
            if random.random() < 0.6:  # Add noise
                noise_level = random.choice(self.augmentation_params['noise_levels'])
                augmented = self.add_noise(augmented, noise_level)
            
            if random.random() < 0.4:  # Pitch shift
                pitch_shift = random.choice(self.augmentation_params['pitch_shifts'])
                augmented = self.pitch_shift(augmented, pitch_shift)
            
            if random.random() < 0.5:  # Speed change
                speed_factor = random.choice(self.augmentation_params['speed_factors'])
                augmented = self.time_stretch(augmented, speed_factor)
            
            if random.random() < 0.3:  # Volume change
                volume_factor = random.choice(self.augmentation_params['volume_factors'])
                augmented = self.volume_change(augmented, volume_factor)
            
            if random.random() < 0.2:  # Add reverb
                augmented = self.add_reverb(augmented)
            
            # Final normalization
            if np.max(np.abs(augmented)) > 0:
                augmented = augmented / (np.max(np.abs(augmented)) + 1e-7)
            augmented_audios.append(augmented)
        
        return augmented_audios
    
    def process_stammering_dataset(self, dataset: List[Dict], max_samples: int = 50) -> Dict:
        """Process the stammering dataset to create synthetic audio with multiple speakers"""
        
        print(f"🎯 Processing {min(len(dataset), max_samples)} dataset entries...")
        
        processed_data = {
            'audio_files': [],
            'labels': [],
            'features': [],
            'metadata': []
        }
        
        for i, entry in enumerate(dataset[:max_samples]):
            print(f"\n📝 Processing {i+1}/{min(len(dataset), max_samples)}: {entry['id']}")
            
            try:
                # Use different speakers for variety
                normal_speaker_id = self.get_random_speaker_id()
                stammer_speaker_id = self.get_random_speaker_id()
                
                # Generate normal speech
                print(f"  🎤 Generating normal speech with speaker {normal_speaker_id}...")
                normal_path = self.output_dir / "raw_synthetic" / f"{entry['id']}_normal.wav"
                normal_audio = self.synthesize_speech(entry['original'], str(normal_path), normal_speaker_id)
                
                # Generate stammered speech
                print(f"  🗣️  Generating stammered speech with speaker {stammer_speaker_id}...")
                stammer_path = self.output_dir / "raw_synthetic" / f"{entry['id']}_stammer.wav"
                stammer_audio = self.synthesize_speech(entry['stammered'], str(stammer_path), stammer_speaker_id)
                
                # Create augmented versions
                print("  🔄 Creating augmented versions...")
                normal_augmented = self.augment_audio(normal_audio, 2)
                stammer_augmented = self.augment_audio(stammer_audio, 3)
                
                # Save augmented normal audio
                for j, aug_audio in enumerate(normal_augmented):
                    aug_path = self.output_dir / "augmented" / f"{entry['id']}_normal_aug_{j}.wav"
                    sf.write(aug_path, aug_audio, self.sample_rate)
                    
                    processed_data['audio_files'].append(str(aug_path))
                    processed_data['labels'].append(0)  # 0 = normal
                    processed_data['metadata'].append({
                        'file': str(aug_path),
                        'text': entry['original'],
                        'type': 'normal',
                        'augmentation': j,
                        'speaker_id': normal_speaker_id,
                        'speaker_name': self.get_speaker_name(normal_speaker_id) if normal_speaker_id is not None else None
                    })
                
                # Save augmented stammered audio
                for j, aug_audio in enumerate(stammer_augmented):
                    aug_path = self.output_dir / "augmented" / f"{entry['id']}_stammer_aug_{j}.wav"
                    sf.write(aug_path, aug_audio, self.sample_rate)
                    
                    processed_data['audio_files'].append(str(aug_path))
                    processed_data['labels'].append(1)  # 1 = stammered
                    processed_data['metadata'].append({
                        'file': str(aug_path),
                        'text': entry['stammered'],
                        'type': 'stammered',
                        'severity': entry['severity'],
                        'patterns': entry['patterns'],
                        'augmentation': j,
                        'speaker_id': stammer_speaker_id,
                        'speaker_name': self.get_speaker_name(stammer_speaker_id) if stammer_speaker_id is not None else None
                    })
                
                print(f"  ✅ Entry {i+1} processed successfully")
                
            except Exception as e:
                print(f"  ❌ Error processing entry {i+1}: {e}")
                continue
        
        # Save metadata
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(processed_data['metadata'], f, indent=2)
        
        # Save speaker info
        if self.available_speakers:
            speaker_info_path = self.output_dir / "speaker_info.json"
            speaker_info = {
                'model_tag': self.model_tag,
                'total_speakers': len(self.available_speakers),
                'spk2id': self.spk2id,
                'id2spk': self.id2spk,
                'available_speakers': self.available_speakers
            }
            with open(speaker_info_path, 'w') as f:
                json.dump(speaker_info, f, indent=2)
            print(f"  📋 Speaker info saved: {speaker_info_path}")
        
        print(f"\n🎉 Processing complete!")
        print(f"  📊 Total audio files: {len(processed_data['audio_files'])}")
        print(f"  📈 Normal samples: {processed_data['labels'].count(0)}")
        print(f"  📉 Stammered samples: {processed_data['labels'].count(1)}")
        print(f"  🎭 Speakers used: {len(self.available_speakers) if self.available_speakers else 1}")
        print(f"  💾 Metadata saved: {metadata_path}")
        
        return processed_data

# Usage example
if __name__ == "__main__":
    print("🚀 ESPnet Multi-Speaker Synthetic Stammering Pipeline")
    print("=" * 60)
    
    # Load your stammering dataset
    try:
        with open('enhanced_stammering_dataset.json', 'r') as f:
            dataset = json.load(f)
        print(f"✅ Loaded dataset with {len(dataset)} entries")
    except FileNotFoundError:
        print("❌ enhanced_stammering_dataset.json not found!")
        print("Please run your stammering dataset generator first.")
        exit(1)
    
    # Initialize synthesizer with multi-speaker model
    print("\n🔧 Setting up Multi-Speaker ESPnet synthesizer...")
    
    # Recommended multi-speaker models in order of preference
    multi_speaker_models = [
        "espnet/kan-bayashi_vctk_tts_train_full_band_multi_spk_vits_raw_phn_tacotron_g-truncated-50b003",  # VCTK VITS
        "espnet/espnet_male_en_vctk_vits",  # Alternative VCTK model
        "espnet/espnet_female_en_vctk_vits"  # Female VCTK model
    ]
    
    synthesizer = None
    for model_tag in multi_speaker_models:
        try:
            synthesizer = ESPnetCMUArcticSynthesizer(model_tag=model_tag)
            if synthesizer.tts_model is not None:
                print(f"✅ Successfully initialized with model: {model_tag}")
                break
        except Exception as e:
            print(f"❌ Failed to initialize with {model_tag}: {e}")
            continue
    
    if synthesizer is None or synthesizer.tts_model is None:
        print("❌ Could not initialize any multi-speaker TTS model.")
        print("Falling back to single speaker with multiple augmentations.")
        synthesizer = ESPnetCMUArcticSynthesizer()
    
    # Test synthesis with multiple speakers
    print("\n🧪 Testing multi-speaker synthesis...")
    test_text = "Hello, this is a test sentence."
    test_stammer = "H-h-hello, this is a t-t-test sentence."
    
    if synthesizer.available_speakers:
        # Test with 3 different speakers
        print("Testing with different speakers:")
        for i in range(min(3, len(synthesizer.available_speakers))):
            print(f"  🎭 Testing with speaker {i}: {synthesizer.get_speaker_name(i)}")
            test_audio = synthesizer.synthesize_speech(test_text, speaker_id=i)
            print(f"    Audio length: {len(test_audio)} samples")
    else:
        print("Single speaker model - testing basic synthesis:")
        test_audio_normal = synthesizer.synthesize_speech(test_text)
        test_audio_stammer = synthesizer.synthesize_speech(test_stammer)
        print(f"  Normal audio length: {len(test_audio_normal)} samples")
        print(f"  Stammer audio length: {len(test_audio_stammer)} samples")
    
    # Process dataset with multiple speakers
    print("\n📊 Processing dataset with multi-speaker support...")
    processed_data = synthesizer.process_stammering_dataset(dataset, max_samples=10)
    
    print("\n🎯 Next steps:")
    print("1. ✅ Multi-speaker synthetic audio generated")
    print("2. 🔄 Ready for Wav2Vec 2.0 training")
    print("3. 📁 Check output folder:", synthesizer.output_dir)
    print("4. 🎭 Multi-speaker diversity achieved for better generalization")