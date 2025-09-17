#!/usr/bin/env python3
"""
Synthetic Stammering Audio Generator (Bark TTS)
Generates stammering samples in multiple accents
"""

import torch
import soundfile as sf
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoProcessor, BarkModel

class StammeringGenerator:
    def __init__(self, output_dir="stammering_samples"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Test stammering text
        self.stammering_text = "I-I-I w-want to go to the st-st-store but I c-c-can't find my k-keys"

        # Accent voices (Bark presets)
        self.accents = {
            "american": "v2/en_speaker_6",
            "british": "v2/en_speaker_9",
            "australian": "v2/en_speaker_1",
            "canadian": "v2/en_speaker_2",
            "scottish": "v2/en_speaker_4",
            "hindi": "v2/en_speaker_3"  # closest Hindi-like preset in Bark
        }

        print("🔄 Loading Bark TTS...")
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = BarkModel.from_pretrained("suno/bark-small")
        print("✅ Bark loaded")

    def generate(self):
        for accent, preset in self.accents.items():
            print(f"\n🎭 Generating {accent} accent...")

            text_with_preset = f"[{preset}] {self.stammering_text}"
            inputs = self.processor(text_with_preset, return_tensors="pt")

            with torch.no_grad():
                audio = self.model.generate(**inputs, do_sample=True)

            audio = audio.cpu().numpy().squeeze()
            filename = self.output_dir / f"stammer_{accent}.wav"
            sf.write(filename, audio, 24000)

            print(f"   ✅ Saved {filename}")

def main():
    generator = StammeringGenerator()
    generator.generate()

if __name__ == "__main__":
    main()
