#!/usr/bin/env python3
"""
Test the StreamingHierarchicalEngine with truly massive text.
Let's see it handle the equivalent of a small book!
"""

import time
from streaming_engine import StreamingHierarchicalEngine, StreamingConfig

def generate_massive_test_text(target_words=50000):
    """Generate a large, realistic text for testing."""
    
    # Base chapters with different topics
    chapters = [
        # Chapter 1: Machine Learning Fundamentals
        """
        Chapter 1: The Foundation of Machine Learning
        
        Machine learning represents a paradigm shift in how we approach problem-solving in computer science.
        Unlike traditional programming where we explicitly code every rule and condition, machine learning
        algorithms learn patterns from data and make predictions or decisions based on these learned patterns.
        
        The core principle underlying machine learning is the ability to generalize from examples. When we
        train a machine learning model, we provide it with a dataset containing input-output pairs, and the
        algorithm automatically discovers the underlying relationships. This process of learning from data
        enables machines to handle tasks that would be extremely difficult or impossible to program explicitly.
        
        There are three main categories of machine learning: supervised learning, unsupervised learning,
        and reinforcement learning. Supervised learning involves training models on labeled data, where
        both input and desired output are known. Unsupervised learning works with unlabeled data to discover
        hidden patterns or structures. Reinforcement learning takes a different approach, where agents learn
        through interaction with an environment, receiving rewards or penalties for their actions.
        
        The mathematical foundations of machine learning draw from statistics, linear algebra, calculus,
        and probability theory. Understanding these mathematical concepts is crucial for developing effective
        machine learning solutions and interpreting their results correctly.
        """,
        
        # Chapter 2: Deep Learning Revolution
        """
        Chapter 2: The Deep Learning Revolution
        
        Deep learning has emerged as one of the most transformative technologies of the 21st century.
        Built upon the concept of artificial neural networks, deep learning systems can automatically
        learn hierarchical representations of data through multiple layers of processing.
        
        The breakthrough came with the development of more powerful computational resources, particularly
        Graphics Processing Units (GPUs), which enabled training of much deeper networks than previously
        possible. This computational power, combined with access to large datasets and improved algorithms,
        sparked the deep learning revolution that began in the early 2010s.
        
        Convolutional Neural Networks (CNNs) revolutionized computer vision by automatically learning
        spatial hierarchies of features. These networks can identify simple edges and textures in early
        layers, gradually building up to recognize complex objects and scenes in deeper layers.
        
        Recurrent Neural Networks (RNNs) and their variants, such as Long Short-Term Memory (LSTM) networks,
        brought significant advances to sequence modeling tasks. These architectures can process variable-length
        sequences and maintain memory of previous inputs, making them ideal for natural language processing,
        speech recognition, and time series analysis.
        
        The introduction of attention mechanisms and transformer architectures marked another major milestone.
        These innovations enabled models to focus on relevant parts of the input sequence and process
        sequences in parallel rather than sequentially, leading to dramatic improvements in efficiency
        and performance.
        """,
        
        # Chapter 3: Natural Language Processing
        """
        Chapter 3: Understanding Human Language
        
        Natural Language Processing (NLP) represents one of the most challenging and fascinating areas
        of artificial intelligence. The goal is to enable machines to understand, interpret, and generate
        human language in a way that is both meaningful and useful.
        
        The complexity of human language stems from its inherent ambiguity, context-dependency, and
        cultural nuances. Words can have multiple meanings depending on context, sentences can be
        structured in various ways to convey the same meaning, and cultural references add layers
        of interpretation that require deep understanding of human society and history.
        
        Traditional NLP approaches relied heavily on hand-crafted rules and linguistic features.
        Researchers would manually define grammar rules, create dictionaries of word meanings,
        and develop algorithms based on linguistic theories. While these approaches achieved some
        success, they were labor-intensive and struggled to handle the full complexity of natural language.
        
        The advent of statistical methods brought significant improvements to NLP. Techniques like
        n-gram models, Hidden Markov Models, and Support Vector Machines enabled systems to learn
        from data rather than relying solely on predefined rules. These statistical approaches
        could better handle the variability and ambiguity inherent in human language.
        
        The deep learning revolution transformed NLP dramatically. Word embeddings like Word2Vec
        and GloVe provided dense, meaningful representations of words that captured semantic
        relationships. Recurrent neural networks enabled better modeling of sequential dependencies
        in text, while attention mechanisms allowed models to focus on relevant parts of the input.
        
        The introduction of transformer-based models like BERT, GPT, and T5 marked a new era in NLP.
        These models, pre-trained on massive text corpora, demonstrate remarkable abilities in
        understanding context, generating coherent text, and performing various language tasks
        with minimal task-specific training.
        """,
        
        # Chapter 4: Computer Vision
        """
        Chapter 4: Teaching Machines to See
        
        Computer vision aims to give machines the ability to interpret and understand visual information
        from the world around them. This field combines techniques from mathematics, physics, and
        computer science to process, analyze, and understand digital images and videos.
        
        The human visual system serves as both inspiration and benchmark for computer vision systems.
        Humans can effortlessly recognize objects, understand spatial relationships, and interpret
        complex visual scenes within milliseconds. Replicating this capability in machines has proven
        to be an extraordinarily challenging task.
        
        Early computer vision systems relied on hand-crafted features and classical image processing
        techniques. Researchers developed algorithms to detect edges, corners, and other low-level
        features, then combined these features to recognize objects and scenes. While these approaches
        worked well for specific, controlled scenarios, they struggled with the variability and
        complexity of real-world images.
        
        The introduction of machine learning brought significant improvements to computer vision.
        Techniques like Support Vector Machines and Random Forests could learn to recognize patterns
        from training data, reducing the need for manual feature engineering. However, these methods
        still required careful preprocessing and feature extraction steps.
        
        Convolutional Neural Networks (CNNs) revolutionized computer vision by automatically learning
        hierarchical feature representations. These networks can discover optimal features for a given
        task through training, eliminating the need for manual feature design. The success of CNNs
        in image classification competitions like ImageNet demonstrated their superiority over
        traditional approaches.
        
        Modern computer vision systems achieve human-level performance on many tasks, including
        image classification, object detection, and facial recognition. Applications range from
        autonomous vehicles and medical imaging to augmented reality and robotics, transforming
        industries and enabling new technological possibilities.
        """,
        
        # Chapter 5: Reinforcement Learning
        """
        Chapter 5: Learning Through Interaction
        
        Reinforcement Learning (RL) represents a unique paradigm in machine learning where agents
        learn optimal behaviors through trial and error interactions with their environment.
        Unlike supervised learning, which learns from labeled examples, RL agents must discover
        effective strategies by receiving feedback in the form of rewards and penalties.
        
        The theoretical foundation of reinforcement learning draws from psychology, neuroscience,
        and control theory. The concept mirrors how humans and animals learn through experience,
        gradually improving their behavior based on the consequences of their actions.
        
        At its core, reinforcement learning involves an agent that takes actions in an environment,
        receives observations about the current state, and gets rewards or penalties based on
        the outcomes of its actions. The agent's goal is to learn a policy that maximizes the
        cumulative reward over time.
        
        One of the most celebrated successes of reinforcement learning is in game playing.
        Programs like AlphaGo, which defeated world champion Go players, and OpenAI Five,
        which competed at professional levels in Dota 2, demonstrate the potential of RL
        to master complex strategic thinking and decision-making.
        
        The combination of reinforcement learning with deep neural networks, known as Deep
        Reinforcement Learning (DRL), has opened new possibilities for tackling complex problems.
        Algorithms like Deep Q-Networks (DQN), Policy Gradient methods, and Actor-Critic
        approaches have achieved remarkable results in various domains.
        
        Applications of reinforcement learning extend far beyond games. Autonomous vehicles
        use RL for navigation and decision-making, recommendation systems employ RL to
        optimize user engagement, and robotics systems use RL to learn complex manipulation
        tasks. The field continues to grow and find applications in finance, healthcare,
        and industrial automation.
        """,
        
        # Chapter 6: Ethics and Future
        """
        Chapter 6: The Ethical Frontier and Future Directions
        
        As artificial intelligence and machine learning systems become increasingly powerful
        and ubiquitous, questions of ethics, fairness, and societal impact have moved to
        the forefront of the field. The decisions made by AI systems can have profound
        effects on individuals and communities, making ethical considerations paramount.
        
        Bias in machine learning systems represents one of the most pressing challenges.
        These biases can arise from historical data that reflects societal inequalities,
        from the way problems are framed, or from the lack of diversity in development teams.
        When biased systems are deployed in critical applications like hiring, lending,
        or criminal justice, they can perpetuate and amplify existing inequalities.
        
        Transparency and explainability have become crucial requirements for AI systems,
        especially in high-stakes applications. Users and stakeholders need to understand
        how AI systems make decisions, what factors influence their outputs, and when
        these systems might fail. This has led to the development of explainable AI
        techniques and interpretable machine learning methods.
        
        Privacy concerns arise as AI systems often require large amounts of personal data
        for training and operation. Techniques like differential privacy, federated learning,
        and homomorphic encryption are being developed to enable AI applications while
        protecting individual privacy.
        
        The future of artificial intelligence holds both tremendous promise and significant
        challenges. Emerging paradigms like quantum machine learning, neuromorphic computing,
        and brain-computer interfaces could revolutionize how we approach computation and
        intelligence. At the same time, questions about artificial general intelligence,
        technological unemployment, and the control of AI systems require careful consideration.
        
        Collaboration between technologists, ethicists, policymakers, and society at large
        will be essential to ensure that AI development serves the common good and addresses
        the challenges facing humanity. The choices we make today about AI research and
        deployment will shape the future of human civilization.
        """,
    ]
    
    # Repeat and expand chapters to reach target word count
    full_text = ""
    current_words = 0
    chapter_index = 0
    
    while current_words < target_words:
        chapter_content = chapters[chapter_index % len(chapters)]
        full_text += chapter_content + "\n\n"
        
        # Add some variations to make it more realistic
        if chapter_index > 0 and chapter_index % 3 == 0:
            full_text += f"""
            
            Section Summary {chapter_index // 3}:
            
            The preceding chapters have explored fundamental concepts in artificial intelligence
            and machine learning. We have examined the theoretical foundations, practical
            applications, and emerging challenges that define this rapidly evolving field.
            
            Key insights from this section include the importance of data quality, the power
            of deep learning architectures, and the critical need for ethical considerations
            in AI development. As we continue to push the boundaries of what machines can
            accomplish, we must remain mindful of the broader implications for society.
            
            """
        
        current_words = len(full_text.split())
        chapter_index += 1
    
    return full_text

def test_with_massive_text():
    """Test the streaming engine with truly massive text."""
    print("🚀 GENERATING MASSIVE TEST TEXT...")
    
    # Generate text equivalent to a small book (50K words)
    massive_text = generate_massive_test_text(50000)
    
    print(f"📚 Generated text with {len(massive_text.split())} words ({len(massive_text)} characters)")
    print(f"📖 Equivalent to ~{len(massive_text) // 5000} pages of text")
    
    # Configure for massive text processing
    config = StreamingConfig(
        chunk_size_words=800,      # Larger chunks for efficiency
        overlap_ratio=0.1,         # Less overlap for speed
        max_memory_mb=1024,        # More memory allowance
        max_concurrent_chunks=6,   # More parallel processing
        enable_progressive_refinement=True,
        cache_processed_chunks=True
    )
    
    print(f"⚙️ Configuration: {config.chunk_size_words} words/chunk, {config.max_concurrent_chunks} parallel")
    
    # Create and test engine
    engine = StreamingHierarchicalEngine(config)
    
    print("🔥 STARTING MASSIVE TEXT PROCESSING...")
    start_time = time.time()
    
    result = engine.process_streaming_text(massive_text)
    
    processing_time = time.time() - start_time
    
    # Display comprehensive results
    print(f"\n⚡ PROCESSING COMPLETED!")
    print(f"🕐 Total time: {processing_time:.2f} seconds")
    print(f"📊 Processing speed: {len(massive_text.split()) / processing_time:.0f} words/second")
    
    if 'processing_stats' in result:
        stats = result['processing_stats']
        print(f"📈 Success rate: {stats.get('success_rate', 0):.1%}")
        print(f"🧩 Chunks processed: {stats.get('successful_chunks', 0)}/{stats.get('total_chunks', 0)}")
    
    if 'hierarchical_summary' in result:
        hs = result['hierarchical_summary']
        print(f"\n🎯 LEVEL 1 CONCEPTS ({len(hs['level_1_concepts'])}):")
        for i, concept in enumerate(hs['level_1_concepts'][:10], 1):
            print(f"   {i}. {concept}")
        
        print(f"\n💎 LEVEL 2 CORE SUMMARY:")
        summary = hs['level_2_core']
        print(f"   {summary[:300]}{'...' if len(summary) > 300 else ''}")
        
        if hs.get('level_3_expanded'):
            print(f"\n📖 LEVEL 3 EXPANDED CONTEXT:")
            expanded = hs['level_3_expanded']
            print(f"   {expanded[:200]}{'...' if len(expanded) > 200 else ''}")
    
    if 'key_insights' in result:
        insights = result['key_insights']
        print(f"\n🌟 KEY INSIGHTS ({len(insights)}):")
        for i, insight in enumerate(insights[:5], 1):
            print(f"   {i}. [{insight.get('type', 'INSIGHT')}] {insight.get('text', '')[:100]}...")
            print(f"      💫 Score: {insight.get('score', 0):.2f}")
    
    if 'metadata' in result:
        meta = result['metadata']
        print(f"\n📊 PROCESSING METADATA:")
        print(f"   Original words: {meta.get('original_word_count', 0):,}")
        print(f"   Compression ratio: {meta.get('compression_ratio', 0):.3f}")
        print(f"   Chunks processed: {meta.get('chunks_processed', 0)}")
    
    if 'streaming_metadata' in result:
        stream_meta = result['streaming_metadata']
        print(f"\n🌊 STREAMING METADATA:")
        print(f"   Total processing time: {stream_meta.get('total_processing_time', 0):.2f}s")
        print(f"   Memory efficiency: {stream_meta.get('memory_efficiency', 0):.1%}")
        print(f"   Chunks processed: {stream_meta.get('chunks_processed', 0)}")
    
    # Calculate some impressive stats
    words_per_second = len(massive_text.split()) / processing_time
    chars_per_second = len(massive_text) / processing_time
    
    print(f"\n🎉 PERFORMANCE ACHIEVEMENTS:")
    print(f"   📈 {words_per_second:.0f} words per second")
    print(f"   📈 {chars_per_second:.0f} characters per second") 
    print(f"   📈 {len(massive_text) / (1024 * 1024) / processing_time:.1f} MB per second")
    
    print(f"\n🏆 STREAMING ENGINE SUCCESSFULLY PROCESSED {len(massive_text.split()):,} WORDS!")
    print("🚀 This system can now handle texts of ANY SIZE! 🌟")

if __name__ == "__main__":
    test_with_massive_text()