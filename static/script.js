document.addEventListener('DOMContentLoaded', () => {
    const summarizeBtn = document.getElementById('summarize-btn');
    const summaryOutput = document.getElementById('summary-output');
    const modelSelect = document.getElementById('model-type');
    
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

    summarizeBtn.addEventListener('click', async () => {
        const text = document.getElementById('input-text').value;
        if (!text) {
            alert('Please enter text to summarize');
            return;
        }

        try {
            summaryOutput.textContent = 'Processing...';
            const response = await fetch('/process_text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    text: text,
                    model: modelSelect.value,
                    config: modelSelect.value === 'tiny' ? tinyLLMConfig : {}
                })
            });

            const result = await response.json();

            if (result.error) {
                summaryOutput.textContent = `Error: ${result.error}`;
            } else {
                summaryOutput.textContent = result.summary || 'No summary generated';
            }
        } catch (error) {
            console.error('Error:', error);
            summaryOutput.textContent = 'An error occurred while processing the text';
        }
    });
});
