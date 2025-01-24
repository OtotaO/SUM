document.addEventListener('DOMContentLoaded', () => {
    const summaryLevels = document.getElementsByName('summary-level');
    const modelType = document.getElementById('model-type');
    const customModel = document.getElementById('custom-model');
    const uploadBtn = document.getElementById('upload-model');
    const summarizeBtn = document.getElementById('summarize-btn');
    const summaryOutput = document.getElementById('summary-output');

    let selectedLevel = 'Sum'; // Default level

    // Add event listeners to radio buttons
    for (let i = 0; i < summaryLevels.length; i++) {
        summaryLevels[i].addEventListener('change', (e) => {
            selectedLevel = e.target.value;
        });
    }


    modelType.addEventListener('change', (e) => {
        if (e.target.value === 'custom') {
            customModel.style.display = 'block';
        } else {
            customModel.style.display = 'none';
        }
    });

    uploadBtn.addEventListener('click', () => {
        customModel.click();
    });

    customModel.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (file) {
            const formData = new FormData();
            formData.append('model', file);
            try {
                const response = await fetch('/upload_model', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                alert(result.message);
            } catch (error) {
                alert('Error uploading model');
            }
        }
    });


    summarizeBtn.addEventListener('click', async () => {
        const text = document.getElementById('input-text').value;
        if (!text) {
            alert('Please enter text to summarize');
            return;
        }

        try {
            summaryOutput.textContent = 'Processing...';
            const response = await fetch('/summarize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text: text,
                    level: selectedLevel, // Use selected radio button value
                    model: modelType.value
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            if (result.error) {
                throw new Error(result.error);
            }

            summaryOutput.textContent = result.summary || result.minimum;

            document.getElementById('compression-ratio').textContent = 
                `Compression: ${result.compression_ratio || 0}%`;
            document.getElementById('processing-time').textContent = 
                `Processing Time: ${result.processing_time || 0}ms`;
        } catch (error) {
            console.error('Error:', error);
            summaryOutput.textContent = `Error: ${error.message}`;
        }
    });
});