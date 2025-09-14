import random
import re
import json
from typing import List, Dict, Tuple

class StammeringDatasetGenerator:
    """
    Enhanced stammering dataset generator with realistic patterns
    for synthetic voice generation and augmentation
    """
    
    def __init__(self):
        # Different types of stammering patterns
        self.repetition_patterns = {
            'sound': ['b-b-b-', 'c-c-c-', 'd-d-d-', 'f-f-f-', 'g-g-g-', 'h-h-h-', 'j-j-j-', 'k-k-k-', 'l-l-l-', 'm-m-m-', 'n-n-n-', 'p-p-p-', 'q-q-q-', 'r-r-r-', 's-s-s-', 't-t-t-', 'v-v-v-', 'w-w-w-', 'x-x-x-', 'y-y-y-', 'z-z-z-'],
            'syllable': ['ba-ba-ba-', 'co-co-co-', 'de-de-de-', 'fi-fi-fi-', 'go-go-go-', 'he-he-he-', 'in-in-in-', 'ja-ja-ja-', 'ka-ka-ka-', 'la-la-la-', 'ma-ma-ma-', 'no-no-no-', 'po-po-po-', 'qu-qu-qu-', 're-re-re-', 'so-so-so-', 'ta-ta-ta-', 'un-un-un-', 'vo-vo-vo-', 'wa-wa-wa-'],
            'word': ['I-I-I ', 'the-the-the ', 'and-and-and ', 'but-but-but ', 'can-can-can ', 'will-will-will ', 'have-have-have ', 'want-want-want ', 'need-need-need ', 'think-think-think ']
        }
        
        # Prolongations (vowel stretching)
        self.prolongations = {
            'a': ['aaa', 'aaaa', 'aaaaa'],
            'e': ['eee', 'eeee', 'eeeee'],
            'i': ['iii', 'iiii', 'iiiii'],
            'o': ['ooo', 'oooo', 'ooooo'],
            'u': ['uuu', 'uuuu', 'uuuuu']
        }
        
        # Blocks (silent pauses)
        self.blocks = ['...', '....', '.....', '......']
        
        # Interjections and fillers
        self.fillers = ['um', 'uh', 'er', 'ah', 'em', 'hmm']
        
    def add_repetition_stammer(self, text: str, intensity: float = 0.3) -> str:
        """Add repetition-based stammering to text"""
        words = text.split()
        modified_words = []
        
        for word in words:
            if random.random() < intensity and len(word) > 2:
                stammer_type = random.choice(['sound', 'syllable', 'word'])
                
                if stammer_type == 'sound':
                    # Repeat first sound
                    first_char = word[0].lower()
                    if first_char.isalpha():
                        repetitions = random.randint(2, 4)
                        stammer = (first_char + '-') * repetitions
                        modified_words.append(stammer + word)
                    else:
                        modified_words.append(word)
                        
                elif stammer_type == 'syllable':
                    # Repeat first syllable
                    if len(word) >= 3:
                        syllable = word[:2]
                        repetitions = random.randint(2, 3)
                        stammer = (syllable + '-') * repetitions
                        modified_words.append(stammer + word)
                    else:
                        modified_words.append(word)
                        
                elif stammer_type == 'word':
                    # Repeat entire word
                    repetitions = random.randint(2, 3)
                    stammer = (word + '-') * repetitions
                    modified_words.append(stammer + word)
            else:
                modified_words.append(word)
                
        return ' '.join(modified_words)
    
    def add_prolongation_stammer(self, text: str, intensity: float = 0.2) -> str:
        """Add prolongation-based stammering to text"""
        for vowel, prolongations in self.prolongations.items():
            if random.random() < intensity:
                pattern = re.compile(vowel, re.IGNORECASE)
                replacement = random.choice(prolongations)
                text = pattern.sub(replacement, text, count=1)
        return text
    
    def add_blocks(self, text: str, intensity: float = 0.15) -> str:
        """Add blocks (silent pauses) to text"""
        words = text.split()
        modified_words = []
        
        for i, word in enumerate(words):
            if random.random() < intensity and i > 0:
                block = random.choice(self.blocks)
                modified_words.append(block + ' ' + word)
            else:
                modified_words.append(word)
                
        return ' '.join(modified_words)
    
    def add_fillers(self, text: str, intensity: float = 0.2) -> str:
        """Add interjections and fillers"""
        words = text.split()
        modified_words = []
        
        for i, word in enumerate(words):
            if random.random() < intensity:
                filler = random.choice(self.fillers)
                modified_words.append(filler + ', ' + word)
            else:
                modified_words.append(word)
                
        return ' '.join(modified_words)
    
    def generate_stammering_variations(self, base_text: str, num_variations: int = 5) -> List[Dict]:
        """Generate multiple stammering variations of the base text"""
        variations = []
        
        for i in range(num_variations):
            # Apply different combinations of stammering patterns
            modified_text = base_text
            
            # Randomly apply different types of stammering
            if random.random() < 0.8:  # High chance for repetitions
                modified_text = self.add_repetition_stammer(modified_text, random.uniform(0.2, 0.5))
            
            if random.random() < 0.4:  # Medium chance for prolongations
                modified_text = self.add_prolongation_stammer(modified_text, random.uniform(0.1, 0.3))
            
            if random.random() < 0.3:  # Lower chance for blocks
                modified_text = self.add_blocks(modified_text, random.uniform(0.05, 0.2))
            
            if random.random() < 0.4:  # Medium chance for fillers
                modified_text = self.add_fillers(modified_text, random.uniform(0.1, 0.25))
            
            # Calculate severity score
            severity = self.calculate_severity(base_text, modified_text)
            
            variations.append({
                'id': f'stammer_var_{i+1}',
                'original': base_text,
                'stammered': modified_text,
                'severity': severity,
                'patterns': self.identify_patterns(modified_text)
            })
            
        return variations
    
    def calculate_severity(self, original: str, stammered: str) -> float:
        """Calculate stammering severity score (0-1)"""
        original_words = len(original.split())
        stammered_length = len(stammered)
        original_length = len(original)
        
        # Simple severity metric based on length increase and pattern complexity
        length_ratio = stammered_length / original_length if original_length > 0 else 1
        severity = min((length_ratio - 1) * 2, 1.0)  # Normalize to 0-1 range
        
        return round(severity, 3)
    
    def identify_patterns(self, text: str) -> List[str]:
        """Identify stammering patterns in text"""
        patterns = []
        
        if re.search(r'[a-z]-[a-z]-', text):
            patterns.append('repetition')
        if re.search(r'[aeiou]{3,}', text):
            patterns.append('prolongation')
        if '...' in text:
            patterns.append('block')
        if any(filler in text.lower() for filler in self.fillers):
            patterns.append('filler')
            
        return patterns

# Base sentences for dataset generation
base_sentences = [
    "Hello, how are you doing today?",
    "I would like to introduce myself to you.",
    "Can you help me with this problem?",
    "I'm trying to explain my thoughts clearly.",
    "This is a very important presentation.",
    "I want to tell you about my experiences.",
    "Please give me a moment to think.",
    "I believe we can solve this together.",
    "Thank you for your patience and understanding.",
    "I'm working hard to improve my communication.",
    "Every person deserves to be heard and understood.",
    "I practice speaking exercises every single day.",
    "My confidence grows stronger with each conversation.",
    "I refuse to let challenges define my worth.",
    "Communication is more than just perfect speech.",
    "I embrace my unique voice and perspective.",
    "Progress happens one word at a time.",
    "I am brave enough to share my story.",
    "My message matters regardless of how I say it.",
    "I celebrate every small victory along the way.",
    "Support from others makes a huge difference.",
    "I will never give up on myself.",
    "My journey teaches me patience and resilience.",
    "I inspire others through my determination.",
    "Speaking is an art that requires practice.",
    "I focus on connection rather than perfection.",
    "My voice carries weight and importance.",
    "I grow stronger through every challenge.",
    "Understanding begins with listening to others.",
    "I choose courage over comfort every day."
]

# Generate enhanced dataset
def generate_enhanced_dataset():
    generator = StammeringDatasetGenerator()
    complete_dataset = []
    
    for i, sentence in enumerate(base_sentences):
        variations = generator.generate_stammering_variations(sentence, num_variations=3)
        
        for variation in variations:
            variation['sentence_id'] = i + 1
            complete_dataset.append(variation)
    
    return complete_dataset

# Generate the dataset
if __name__ == "__main__":
    dataset = generate_enhanced_dataset()
    
    # Save to JSON file
    with open('enhanced_stammering_dataset.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    # Print sample entries
    print("Sample dataset entries:")
    print("=" * 50)
    for i, entry in enumerate(dataset[:5]):
        print(f"Entry {i+1}:")
        print(f"Original: {entry['original']}")
        print(f"Stammered: {entry['stammered']}")
        print(f"Severity: {entry['severity']}")
        print(f"Patterns: {', '.join(entry['patterns'])}")
        print("-" * 30)
    
    print(f"\nTotal dataset size: {len(dataset)} entries")
    
    # Statistics
    severities = [entry['severity'] for entry in dataset]
    avg_severity = sum(severities) / len(severities)
    print(f"Average severity: {avg_severity:.3f}")
    
    pattern_counts = {}
    for entry in dataset:
        for pattern in entry['patterns']:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
    
    print("Pattern distribution:")
    for pattern, count in sorted(pattern_counts.items()):
        print(f"  {pattern}: {count} occurrences")