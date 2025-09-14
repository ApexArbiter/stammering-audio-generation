import os
import torch
import soundfile as sf
from espnet2.bin.tts_inference import Text2Speech
import random

# ---------- SETTINGS ----------
NORMAL_SENTENCE = "Hello, my name is Mahad and I am working on speech synthesis."
STAMMER_SENTENCE = "H-h-hello, m-my name is M-mahad and I... I am w-working on s-speech synthesis."
OUTPUT_DIR = "tts_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- CHOOSE MODEL ----------
# Multi-speaker VCTK model (full path with proper model)
model_tag = "espnet/kan-bayashi_vctk_tts_train_full_band_multi_spk_vits_raw_phn_tacotron_g-truncated-50b003"

# Alternative models to try if the above fails
fallback_models = [
    "espnet/kan-bayashi_vctk_vits",  # Your current choice
    "espnet/kan-bayashi_ljspeech_vits",  # Single speaker fallback
]

# ---------- LOAD MODEL ----------
def load_model_safely(model_tags):
    """Try to load models in order of preference"""
    for model_tag in model_tags:
        try:
            print(f"🔄 Attempting to load: {model_tag}")
            text2speech = Text2Speech.from_pretrained(model_tag)
            print(f"✅ Successfully loaded: {model_tag}")
            return text2speech, model_tag
        except Exception as e:
            print(f"❌ Failed to load {model_tag}: {e}")
            continue
    raise Exception("Could not load any model")

# Try to load the best model first, then fallbacks
all_models = [model_tag] + fallback_models
text2speech, loaded_model = load_model_safely(all_models)

# ---------- DETECT SPEAKERS ----------
def get_speaker_info(model):
    """Get speaker information with multiple detection methods"""
    speakers = []
    spk2id = {}
    
    # Method 1: Direct spk2id attribute
    if hasattr(model, "spk2id") and model.spk2id is not None:
        spk2id = model.spk2id
        speakers = list(spk2id.items())
        print(f"✅ Found speakers via model.spk2id: {len(speakers)} speakers")
        
    # Method 2: Check in the TTS generator
    elif hasattr(model, 'tts') and hasattr(model.tts, 'generator'):
        if hasattr(model.tts.generator, 'spk2id') and model.tts.generator.spk2id is not None:
            spk2id = model.tts.generator.spk2id
            speakers = list(spk2id.items())
            print(f"✅ Found speakers via generator.spk2id: {len(speakers)} speakers")
    
    # Method 3: Check TTS model attributes
    elif hasattr(model, 'tts') and hasattr(model.tts, 'spk_embed_dim'):
        # Multi-speaker model but no explicit mapping
        if 'vctk' in loaded_model.lower():
            # Create VCTK speaker mapping
            vctk_speakers = [f"p{225 + i}" for i in range(20)]  # First 20 VCTK speakers
            spk2id = {spk: i for i, spk in enumerate(vctk_speakers)}
            speakers = list(spk2id.items())
            print(f"✅ Created VCTK speaker mapping: {len(speakers)} speakers")
        else:
            # Generic multi-speaker
            num_speakers = min(10, getattr(model.tts, 'spk_embed_dim', 10))
            generic_speakers = [f"speaker_{i:03d}" for i in range(num_speakers)]
            spk2id = {spk: i for i, spk in enumerate(generic_speakers)}
            speakers = list(spk2id.items())
            print(f"✅ Created generic speaker mapping: {len(speakers)} speakers")
    
    # Method 4: Single speaker
    else:
        speakers = [(None, None)]
        print("✅ Single speaker model detected")
    
    return speakers, spk2id

speakers, spk2id = get_speaker_info(text2speech)

print(f"\n📊 Model Info:")
print(f"  Model: {loaded_model}")
print(f"  Sample Rate: {text2speech.fs}")
print(f"  Speakers: {len(speakers) if speakers[0][0] is not None else 1}")

if len(speakers) > 1 and speakers[0][0] is not None:
    print(f"  Example speakers: {speakers[:5]}")

# ---------- GENERATE AUDIO ----------
def synthesize_with_error_handling(model, text, speaker_id=None, speaker_name="default"):
    """Synthesize audio with proper error handling"""
    try:
        # Prepare synthesis arguments
        synthesis_kwargs = {"text": text}
        
        # Add speaker ID if available
        if speaker_id is not None:
            # Convert to tensor with proper format
            if isinstance(speaker_id, int):
                speaker_tensor = torch.tensor([speaker_id], dtype=torch.long)
            else:
                speaker_tensor = torch.tensor([int(speaker_id)], dtype=torch.long)
            
            synthesis_kwargs["sids"] = speaker_tensor
            print(f"  🎭 Using speaker '{speaker_name}' (ID: {speaker_id})")
        else:
            print(f"  🎤 Using single speaker")
        
        # Generate audio
        with torch.no_grad():
            output = model(**synthesis_kwargs)
        
        # Extract waveform
        if isinstance(output, dict):
            wav = output["wav"]
        else:
            wav = output
        
        # Convert to numpy if needed
        if hasattr(wav, 'detach'):
            wav = wav.detach().cpu().numpy()
        elif hasattr(wav, 'numpy'):
            wav = wav.numpy()
        
        return wav
        
    except Exception as e:
        print(f"  ❌ Synthesis error: {e}")
        # Try without speaker ID as fallback
        if speaker_id is not None:
            print(f"  🔄 Retrying without speaker ID...")
            return synthesize_with_error_handling(model, text, None, "fallback")
        return None

# Test with limited number of speakers for debugging
max_speakers_to_test = 3 if len(speakers) > 3 else len(speakers)
selected_speakers = speakers[:max_speakers_to_test]

print(f"\n🎵 Generating audio for {max_speakers_to_test} speakers...")

for i, (spk_name, spk_id) in enumerate(selected_speakers):
    speaker_name = spk_name if spk_name else "default"
    print(f"\n🎭 Speaker {i+1}/{max_speakers_to_test}: {speaker_name}")

    for version, sentence in [("normal", NORMAL_SENTENCE), ("stammer", STAMMER_SENTENCE)]:
        print(f"  📝 Generating {version} speech...")
        
        # Synthesize audio
        wav = synthesize_with_error_handling(text2speech, sentence, spk_id, speaker_name)
        
        if wav is not None:
            # Save audio
            out_path = os.path.join(OUTPUT_DIR, f"{speaker_name}_{version}.wav")
            
            try:
                # Ensure proper audio format
                if wav.ndim > 1:
                    wav = wav.flatten()
                
                # Normalize if needed
                if wav.max() > 1.0 or wav.min() < -1.0:
                    wav = wav / (max(abs(wav.max()), abs(wav.min())) + 1e-7)
                
                sf.write(out_path, wav, text2speech.fs, "PCM_16")
                print(f"  ✅ Saved: {out_path} (length: {len(wav)} samples)")
                
            except Exception as e:
                print(f"  ❌ Save error: {e}")
        else:
            print(f"  ❌ Could not generate audio for {speaker_name} {version}")

print(f"\n🎉 Audio generation complete!")
print(f"📁 Check outputs in: {OUTPUT_DIR}")

# ---------- VERIFY OUTPUTS ----------
print(f"\n🔍 Verifying generated files:")
for file in os.listdir(OUTPUT_DIR):
    if file.endswith('.wav'):
        file_path = os.path.join(OUTPUT_DIR, file)
        try:
            audio, sr = sf.read(file_path)
            duration = len(audio) / sr
            print(f"  📄 {file}: {duration:.2f}s, {sr}Hz, {len(audio)} samples")
        except Exception as e:
            print(f"  ❌ Error reading {file}: {e}")