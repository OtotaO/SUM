document.addEventListener('DOMContentLoaded', () => {
    const elements = {
        summarizeBtn: document.getElementById('summarize-btn'),
        summaryOutput: document.getElementById('summary-output'),
        modelSelect: document.getElementById('model-type'),
        inputText: document.getElementById('input-text'),
        tokenCount: document.getElementById('current-token-count')
    };

    const config = {
        tinyLLM: {
            temperature: 0.7,
            maxTokens: 100,
            topP: 0.9,
            frequencyPenalty: 0.0,
            presencePenalty: 0.0
        }
    };

    function updateTokenCount() {
        const words = elements.inputText.value.trim().split(/\s+/);
        const estimatedTokens = Math.ceil(words.length * 1.3);
        elements.tokenCount.textContent = estimatedTokens;
    }

    async function processSummary() {
        try {
            const text = elements.inputText.value.trim();
            if (!text) {
                throw new Error('Please enter text to summarize');
            }

            elements.summaryOutput.textContent = 'Processing...';
            const response = await fetch('/process_text', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text,
                    model: elements.modelSelect.value,
                    config: elements.modelSelect.value === 'tiny' ? config.tinyLLM : {}
                })
            });

            const result = await response.json();
            if (result.error) {
                throw new Error(result.error);
            }

            elements.summaryOutput.textContent = result.summary || 'No summary generated';
        } catch (error) {
            elements.summaryOutput.textContent = `Error: ${error.message}`;
            console.error('Processing error:', error);
        }
    }

    // Event Listeners
    elements.inputText.addEventListener('input', updateTokenCount);
    elements.summarizeBtn.addEventListener('click', processSummary);
});

function updateTinyLLMConfig(param, value) {
    const numValue = parseFloat(value);
    const paramGroup = document.querySelector(`input[onchange*="${param}"]`)
        .closest('.param-group');
    paramGroup.querySelector('.param-value').textContent = numValue;
}