document.addEventListener('DOMContentLoaded', () => {
    const summarizeBtn = document.getElementById('summarize-btn');
    const summaryOutput = document.getElementById('summary-output');
    const modelSelect = document.getElementById('model-type');
    const inputText = document.getElementById('input-text');
    const tokenCount = document.getElementById('current-token-count');
    
    inputText.addEventListener('input', () => {
        const words = inputText.value.trim().split(/\s+/);
        const estimatedTokens = Math.ceil(words.length * 1.3);
        tokenCount.textContent = estimatedTokens;
    });
    
    // TinyLLM Configuration
    const tinyLLMConfig = {
        temperature: 0.7,
        maxTokens: 100,
        topP: 0.9,
        frequencyPenalty: 0.0,
        presencePenalty: 0.0
    };

    function updateTinyLLMConfig(param, value) {
        tinyLLMConfig[param] = parseFloat(value);
    }

    summarizeBtn.addEventListener('click', async () => {
        const text = document.getElementById('input-text').value;
        if (!text) {
            alert('Please enter text to summarize');
            return;
        }

        try {
            summarizeBtn.disabled = true;
            summaryOutput.textContent = 'Processing...';
            
            const response = await fetch('/process_text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    text: text,
                    model: modelSelect.value,
                    config: tinyLLMConfig
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }
            
            summaryOutput.textContent = result.summary || 'No summary generated';
        } catch (error) {
            console.error('Processing error:', error);
            summaryOutput.textContent = `Error: ${error.message}`;
        } finally {
            summarizeBtn.disabled = false;
        }
    });
});
// TinyLLM Configuration
const tinyLLMConfig = {
    temperature: 0.7,
    maxTokens: 100,
    topP: 0.9,
    frequencyPenalty: 0.0,
    presencePenalty: 0.0
};

function updateTinyLLMConfig(param, value) {
    const numValue = parseFloat(value);
    tinyLLMConfig[param] = numValue;
    
    // Update displayed value
    const paramGroup = document.querySelector(`input[onchange*="${param}"]`).closest('.param-group');
    paramGroup.querySelector('.param-value').textContent = numValue;
}
