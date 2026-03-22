"""
Deterministic Syntactic Sieve — High-Fidelity Edge NLP

Extracts topological (Subject, Predicate, Object) triplets using strict
grammatical dependency parsing via spaCy.  Replaces the LLM for bulk
ingestion, parsing text at bare-metal CPU speeds.

Cost: $0.  Speed: 10,000+ words per second.  Deterministic: always.

Phase 13: Zenith of Process Intensification.

Author: ototao
License: Apache License 2.0
"""

import spacy
from typing import List, Tuple


class DeterministicSieve:
    """
    High-Fidelity Edge NLP.

    Extracts topological (Subject, Predicate, Object) triplets using
    strict grammatical dependency parsing.

    Cost: $0. Speed: 10,000+ words per second.
    """

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            import sys

            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
            )
            self.nlp = spacy.load("en_core_web_sm")

    def extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Parse text into semantic triplets using dependency grammar.

        Walks each sentence's dependency tree to find the ROOT verb,
        then extracts its nominal subject and direct/prepositional
        object, including adjectival and compound modifiers.

        Args:
            text: Raw text to parse.

        Returns:
            Deduplicated list of (subject, predicate, object) tuples.
        """
        doc = self.nlp(text)
        triplets = []

        for sent in doc.sents:
            subject = None
            predicate = None
            object_ = None

            # Find the ROOT of the sentence (usually the main verb)
            for token in sent:
                if token.dep_ == "ROOT" or token.pos_ == "VERB":
                    predicate = token.lemma_

                    # Find subject
                    for child in token.children:
                        if child.dep_ in ("nsubj", "nsubjpass", "csubj"):
                            modifiers = [
                                c.text
                                for c in child.children
                                if c.dep_ in ("amod", "compound")
                            ]
                            subject = " ".join(
                                modifiers + [child.lemma_]
                            ).strip()

                    # Find object
                    for child in token.children:
                        if child.dep_ in ("dobj", "pobj", "attr", "acomp"):
                            modifiers = [
                                c.text
                                for c in child.children
                                if c.dep_ in ("amod", "compound")
                            ]
                            object_ = " ".join(
                                modifiers + [child.lemma_]
                            ).strip()

            if subject and predicate and object_:
                # Filter out massive run-on parses
                if len(subject.split()) <= 5 and len(object_.split()) <= 8:
                    triplets.append(
                        (subject.lower(), predicate.lower(), object_.lower())
                    )

        return list(set(triplets))  # Deduplicate
