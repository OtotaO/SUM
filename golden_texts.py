"""
Golden Texts Collection - Incompressible Prose Benchmarks

A curated collection of texts that represent the pinnacle of human
expression - philosophical insights, technical precision, and literary
beauty that resist compression while maintaining meaning.

These texts serve as benchmarks for our adaptive compression engine,
helping us understand the limits of semantic compression.

"Some things cannot be compressed without losing their essence."
- The eternal tension between brevity and meaning

Author: ototao & Claude
License: Apache 2.0
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from adaptive_compression import ContentType


@dataclass
class GoldenText:
    """Represents a benchmark text with metadata."""
    title: str
    author: str
    content: str
    content_type: ContentType
    year: int
    source: str
    incompressibility_score: float  # 0-1, higher = more incompressible
    notes: str = ""


class GoldenTextsCollection:
    """
    Collection of incompressible texts for benchmarking compression algorithms.
    
    These texts represent the theoretical limits of semantic compression -
    where every word carries essential meaning that cannot be reduced
    without significant loss of information.
    """
    
    def __init__(self):
        self.texts = self._initialize_collection()
    
    def _initialize_collection(self) -> Dict[str, List[GoldenText]]:
        """Initialize the collection with curated texts."""
        
        texts = {
            'philosophical': [
                GoldenText(
                    title="Meditations - Book 2, Passage 5",
                    author="Marcus Aurelius",
                    content="""At dawn, when you have trouble getting out of bed, tell yourself: "I have to go to work — as a human being. What do I have to complain of, if I'm going to do what I was born for — the things I was brought into the world to do? Or is this what I was created for? To huddle under the blankets and stay warm?" So you were born to work together with others. Were you born to feel "nice"? Instead of doing things and experiencing them? Don't you see the plants, the birds, the ants and spiders and bees going about their individual tasks, putting the world in order, as best they can? And you're not willing to do your job as a human being? Why aren't you running to do what your nature demands? You don't love yourself enough. Or you'd love your nature too.""",
                    content_type=ContentType.PHILOSOPHICAL,
                    year=161,
                    source="Meditations, Book 2",
                    incompressibility_score=0.95,
                    notes="Every sentence builds essential meaning about duty, nature, and human purpose"
                ),
                
                GoldenText(
                    title="Tao Te Ching - Chapter 1",
                    author="Lao Tzu",
                    content="""The Tao that can be spoken is not the eternal Tao. The name that can be named is not the eternal name. The nameless is the beginning of heaven and earth. The named is the mother of ten thousand things. Ever desireless, one can see the mystery. Ever desiring, one can see the manifestations. These two spring from the same source but differ in name; this appears as darkness. Darkness within darkness. The gate to all mystery.""",
                    content_type=ContentType.PHILOSOPHICAL,
                    year=-600,
                    source="Tao Te Ching",
                    incompressibility_score=0.98,
                    notes="Paradoxical structure where each phrase is essential for the whole meaning"
                ),
                
                GoldenText(
                    title="The Allegory of the Cave",
                    author="Plato",
                    content="""And now, I said, let me show in a figure how far our nature is enlightened or unenlightened: Behold! human beings living in an underground den, which has a mouth open towards the light and reaching all along the den; here they have been from their childhood, and have their legs and necks chained so that they cannot move, and can only see before them, being prevented by the chains from turning round their heads. Above and behind them a fire is blazing at a distance, and between the fire and the prisoners there is a raised way; and you will see, if you look, a low wall built along the way, like the screen which marionette players have in front of them, over which they show the puppets.""",
                    content_type=ContentType.PHILOSOPHICAL,
                    year=-380,
                    source="The Republic, Book VII",
                    incompressibility_score=0.92,
                    notes="Metaphorical precision where imagery conveys philosophical concepts"
                )
            ],
            
            'technical': [
                GoldenText(
                    title="Quicksort Algorithm Description",
                    author="Tony Hoare",
                    content="""Quicksort is a divide-and-conquer algorithm. It works by selecting a 'pivot' element from the array and partitioning the other elements into two sub-arrays, according to whether they are less than or greater than the pivot. The sub-arrays are then sorted recursively. This can be done in-place, requiring small additional amounts of memory to perform the sorting. The algorithm has average-case time complexity of O(n log n) and worst-case of O(n²). However, the worst case occurs rarely in practice with proper pivot selection strategies.""",
                    content_type=ContentType.TECHNICAL,
                    year=1960,
                    source="Computer Journal",
                    incompressibility_score=0.89,
                    notes="Each technical term and complexity notation is essential for understanding"
                ),
                
                GoldenText(
                    title="Unix Philosophy",
                    author="Ken Thompson & Dennis Ritchie",
                    content="""Write programs that do one thing and do it well. Write programs to work together. Write programs to handle text streams, because that is a universal interface. Make each program a filter. Expect the output of every program to become the input to another, as yet unknown, program. Don't clutter output with extraneous information. Avoid stringently columnar or binary input formats. Don't insist on interactive input.""",
                    content_type=ContentType.TECHNICAL,
                    year=1978,
                    source="The Unix Programming Environment",
                    incompressibility_score=0.94,
                    notes="Fundamental principles that define an entire computing paradigm"
                ),
                
                GoldenText(
                    title="CAP Theorem",
                    author="Eric Brewer",
                    content="""In a distributed computer system, you can only simultaneously provide two of the following three guarantees: Consistency (every read receives the most recent write or an error), Availability (every request receives a response, without guarantee that it contains the most recent version of the information), and Partition tolerance (the system continues to operate despite an arbitrary number of messages being dropped or delayed by the network between nodes).""",
                    content_type=ContentType.TECHNICAL,
                    year=2000,
                    source="PODC Keynote",
                    incompressibility_score=0.91,
                    notes="Precise definition of fundamental distributed systems constraint"
                )
            ],
            
            'literary': [
                GoldenText(
                    title="For sale: baby shoes, never worn",
                    author="Attributed to Ernest Hemingway",
                    content="""For sale: baby shoes, never worn.""",
                    content_type=ContentType.NARRATIVE,
                    year=1920,
                    source="Urban legend/Writing exercise",
                    incompressibility_score=0.99,
                    notes="Complete story in six words - ultimate compression already achieved"
                ),
                
                GoldenText(
                    title="Sonnet 18 - Opening",
                    author="William Shakespeare",
                    content="""Shall I compare thee to a summer's day? Thou art more lovely and more temperate: Rough winds do shake the darling buds of May, And summer's lease hath all too short a date.""",
                    content_type=ContentType.POETIC,
                    year=1609,
                    source="Shakespeare's Sonnets",
                    incompressibility_score=0.96,
                    notes="Meter, rhyme, and meaning interwoven inseparably"
                ),
                
                GoldenText(
                    title="The Road Not Taken - Final Stanza",
                    author="Robert Frost",
                    content="""I shall be telling this with a sigh Somewhere ages and ages hence: Two roads diverged in a wood, and I— I took the one less traveled by, And that has made all the difference.""",
                    content_type=ContentType.POETIC,
                    year=1916,
                    source="Mountain Interval",
                    incompressibility_score=0.94,
                    notes="Every pause and repetition carries meaning about choice and regret"
                )
            ],
            
            'mathematical': [
                GoldenText(
                    title="Euler's Identity",
                    author="Leonhard Euler",
                    content="""e^(iπ) + 1 = 0. This equation connects five fundamental mathematical constants: e (Euler's number), i (the imaginary unit), π (Pi), 1 (multiplicative identity), and 0 (additive identity). It demonstrates the profound relationship between exponential and trigonometric functions through complex analysis.""",
                    content_type=ContentType.TECHNICAL,
                    year=1748,
                    source="Introduction to Analysis of the Infinite",
                    incompressibility_score=0.97,
                    notes="Most beautiful equation in mathematics - every symbol essential"
                ),
                
                GoldenText(
                    title="Gödel's Incompleteness Theorem Statement",
                    author="Kurt Gödel",
                    content="""Any consistent formal system F within which a certain amount of elementary arithmetic can be carried out is incomplete; i.e., there are statements of the language of F which can neither be proved nor disproved in F. Furthermore, for any consistent system F, the consistency of F cannot be proved within F itself.""",
                    content_type=ContentType.TECHNICAL,
                    year=1931,
                    source="Über formal unentscheidbare Sätze",
                    incompressibility_score=0.95,
                    notes="Precise logical statement that shattered foundations of mathematics"
                )
            ],
            
            'code': [
                GoldenText(
                    title="Hello World in C",
                    author="Brian Kernighan & Dennis Ritchie",
                    content="""#include <stdio.h>

int main() {
    printf("hello, world\\n");
    return 0;
}""",
                    content_type=ContentType.TECHNICAL,
                    year=1972,
                    source="The C Programming Language",
                    incompressibility_score=0.85,
                    notes="Minimal viable program - pedagogical perfection"
                ),
                
                GoldenText(
                    title="Recursive Factorial",
                    author="Unknown",
                    content="""def factorial(n):
    return 1 if n <= 1 else n * factorial(n - 1)""",
                    content_type=ContentType.TECHNICAL,
                    year=1960,
                    source="Computer Science",
                    incompressibility_score=0.88,
                    notes="Elegant recursive definition mirroring mathematical concept"
                )
            ],
            
            'wisdom': [
                GoldenText(
                    title="Serenity Prayer",
                    author="Reinhold Niebuhr",
                    content="""God, grant me the serenity to accept the things I cannot change, courage to change the things I can, and wisdom to know the difference.""",
                    content_type=ContentType.PHILOSOPHICAL,
                    year=1943,
                    source="Prayer",
                    incompressibility_score=0.93,
                    notes="Complete philosophy of action in one sentence"
                ),
                
                GoldenText(
                    title="Einstein on Simplicity",
                    author="Albert Einstein",
                    content="""Everything should be made as simple as possible, but not simpler.""",
                    content_type=ContentType.PHILOSOPHICAL,
                    year=1946,
                    source="Reader's Digest (paraphrased)",
                    incompressibility_score=0.96,
                    notes="Captures the essence of design philosophy in minimal words"
                ),
                
                GoldenText(
                    title="Wittgenstein's Limits",
                    author="Ludwig Wittgenstein",
                    content="""Whereof one cannot speak, thereof one must be silent.""",
                    content_type=ContentType.PHILOSOPHICAL,
                    year=1921,
                    source="Tractus Logico-Philosophicus",
                    incompressibility_score=0.98,
                    notes="Ultimate statement about the boundaries of language and knowledge"
                )
            ]
        }
        
        return texts
    
    def get_texts_by_type(self, content_type: ContentType) -> List[GoldenText]:
        """Get all texts of a specific content type."""
        all_texts = []
        for category_texts in self.texts.values():
            all_texts.extend([t for t in category_texts if t.content_type == content_type])
        return all_texts
    
    def get_texts_by_category(self, category: str) -> List[GoldenText]:
        """Get all texts in a specific category."""
        return self.texts.get(category, [])
    
    def get_most_incompressible(self, limit: int = 10) -> List[GoldenText]:
        """Get the most incompressible texts across all categories."""
        all_texts = []
        for category_texts in self.texts.values():
            all_texts.extend(category_texts)
        
        return sorted(all_texts, key=lambda t: t.incompressibility_score, reverse=True)[:limit]
    
    def get_benchmark_suite(self) -> Dict[str, List[GoldenText]]:
        """Get a balanced benchmark suite for compression testing."""
        return {
            'philosophical': self.texts['philosophical'][:2],
            'technical': self.texts['technical'][:2], 
            'literary': self.texts['literary'][:2],
            'mathematical': self.texts['mathematical'][:1],
            'code': self.texts['code'][:1],
            'wisdom': self.texts['wisdom'][:2]
        }
    
    def analyze_compression_resistance(self, text: GoldenText, 
                                     compression_result: dict) -> dict:
        """
        Analyze how well a compression preserved the essence of a golden text.
        Returns quality metrics specific to incompressible content.
        """
        original = text.content
        compressed = compression_result.get('compressed', '')
        
        # Basic metrics
        compression_ratio = compression_result.get('actual_ratio', 0)
        
        # Preserve key phrases analysis
        key_phrases = self._extract_key_phrases(original)
        preserved_phrases = sum(1 for phrase in key_phrases if phrase in compressed)
        phrase_preservation = preserved_phrases / max(1, len(key_phrases))
        
        # Semantic coherence (simplified - could use embeddings)
        coherence_score = 0.8  # Placeholder
        
        # Incompressibility violation score
        # Higher score = compression violated the text's incompressible nature
        expected_limit = 1.0 - text.incompressibility_score
        violation_score = max(0, expected_limit - compression_ratio) / expected_limit
        
        return {
            'compression_ratio': compression_ratio,
            'phrase_preservation': phrase_preservation,
            'semantic_coherence': coherence_score,
            'incompressibility_violation': violation_score,
            'quality_score': (phrase_preservation + coherence_score + (1 - violation_score)) / 3,
            'recommended_limit': expected_limit,
            'text_title': text.title,
            'author': text.author
        }
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases that should be preserved in compression."""
        # Simplified implementation - could use NLP
        sentences = text.split('.')
        key_phrases = []
        
        for sentence in sentences:
            words = sentence.strip().split()
            if len(words) > 3:
                # Extract meaningful phrases
                for i in range(len(words) - 2):
                    phrase = ' '.join(words[i:i+3])
                    if len(phrase) > 10:  # Meaningful length
                        key_phrases.append(phrase)
        
        return key_phrases[:5]  # Top 5 key phrases


# Usage example and testing
if __name__ == "__main__":
    collection = GoldenTextsCollection()
    
    print("=== Golden Texts Collection ===\n")
    
    # Show most incompressible texts
    most_incompressible = collection.get_most_incompressible(5)
    print("Top 5 Most Incompressible Texts:")
    for i, text in enumerate(most_incompressible, 1):
        print(f"{i}. {text.title} by {text.author} (Score: {text.incompressibility_score})")
        print(f"   Content: {text.content[:100]}...")
        print()
    
    # Show benchmark suite
    benchmark = collection.get_benchmark_suite()
    print("Benchmark Suite:")
    for category, texts in benchmark.items():
        print(f"\n{category.title()}:")
        for text in texts:
            print(f"  - {text.title} by {text.author}")
    
    # Analyze a compression example
    from adaptive_compression import AdaptiveCompressionEngine
    
    engine = AdaptiveCompressionEngine()
    test_text = collection.texts['philosophical'][0]  # Marcus Aurelius
    
    compression_result = engine.compress(test_text.content, target_ratio=0.5)
    analysis = collection.analyze_compression_resistance(test_text, compression_result)
    
    print(f"\n=== Compression Analysis: {test_text.title} ===")
    print(f"Original ({test_text.incompressibility_score} incompressibility):")
    print(test_text.content[:200] + "...")
    print(f"\nCompressed ({compression_result['actual_ratio']:.1%} ratio):")
    print(compression_result['compressed'][:200] + "...")
    print(f"\nQuality Metrics:")
    print(f"  Phrase Preservation: {analysis['phrase_preservation']:.1%}")
    print(f"  Semantic Coherence: {analysis['semantic_coherence']:.1%}")
    print(f"  Incompressibility Violation: {analysis['incompressibility_violation']:.1%}")
    print(f"  Overall Quality: {analysis['quality_score']:.1%}")
    
    if analysis['incompressibility_violation'] > 0.3:
        print(f"\n⚠️  WARNING: Compression may have violated text's incompressible nature!")
        print(f"   Recommended limit: {analysis['recommended_limit']:.1%}")
    else:
        print(f"\n✅ Compression respects text's incompressible nature")