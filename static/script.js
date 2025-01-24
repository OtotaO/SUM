
document.addEventListener('DOMContentLoaded', () => {
    const summaryLevels = document.getElementsByName('summary-level');
    const modelType = document.getElementById('model-type');
    const summarizeBtn = document.getElementById('summarize-btn');
    const summaryOutput = document.getElementById('summary-output');

    let selectedLevel = 'tags'; // Default level

    // Add event listeners to radio buttons
    for (let i = 0; i < summaryLevels.length; i++) {
        summaryLevels[i].addEventListener('change', (e) => {
            selectedLevel = e.target.id; // Use the id as the level (tags, sum, summary)
        });
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
                    level: selectedLevel,
                    model: 'tiny'
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }

            // Display the appropriate summary based on the selected level
            if (selectedLevel === 'tags') {
                summaryOutput.textContent = result.tags || '';
            } else if (selectedLevel === 'sum') {
                summaryOutput.textContent = result.minimum_summary || '';
            } else {
                summaryOutput.textContent = result.full_summary || '';
            }

            // Update metrics if available
            if (document.getElementById('compression-ratio')) {
                document.getElementById('compression-ratio').textContent = 
                    `Compression: ${result.compression_ratio || 0}%`;
            }
            if (document.getElementById('processing-time')) {
                document.getElementById('processing-time').textContent = 
                    `Processing Time: ${result.processing_time || 0}ms`;
            }
        } catch (error) {
            console.error('Error:', error);
            summaryOutput.textContent = `Error: ${error.message}`;
        }
    });
});
