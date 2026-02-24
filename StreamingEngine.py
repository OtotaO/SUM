from unlimited_text_processor import UnlimitedTextProcessor

class StreamingHierarchicalEngine(UnlimitedTextProcessor):
    def process_streaming_text(self, text, config=None):
        return self.process_text(text, config)
