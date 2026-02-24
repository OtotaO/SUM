import pytest
import sys
import os

# Add parent directory to path so we can import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_rag_crystallizer import GraphRAGCrystallizer

def test_graph_rag_query_focus():
    documents = [
        "Apple Inc. is developing new AI features for the iPhone. Tim Cook announced the partnership with OpenAI.",
        "Microsoft and OpenAI are collaborating on Azure cloud services. Satya Nadella sees AI as transformative.",
        "Google's Gemini model competes with OpenAI's GPT series. Sundar Pichai emphasized multimodal capabilities.",
        "Meta's Llama models are open source. Mark Zuckerberg believes in open AI development."
    ]

    crystallizer = GraphRAGCrystallizer()
    query = "What is Microsoft doing?"

    result = crystallizer.crystallize_corpus(documents, query=query)

    # Assertions
    assert "Microsoft" in result.global_summary, "Summary should mention Microsoft"

    # Check that it filtered down to relevant communities
    # The summary should not be the default "No significant patterns..."
    assert result.global_summary != "No significant patterns found in corpus."

    # Check that duplication fix works (roughly)
    # Count occurrences of "Microsoft (ORG)"
    count = result.global_summary.count("- Microsoft (ORG)")
    # Should be 1, or at least not 4 like before
    assert count <= 2, f"Too many duplicate entries found: {count}"

def test_graph_rag_no_query():
    documents = [
        "Apple Inc. is developing new AI features for the iPhone. Tim Cook announced the partnership with OpenAI.",
        "Microsoft and OpenAI are collaborating on Azure cloud services. Satya Nadella sees AI as transformative."
    ]

    crystallizer = GraphRAGCrystallizer()
    result = crystallizer.crystallize_corpus(documents)

    assert result.global_summary.startswith("Key insights:"), "Should use default prefix when no query"
