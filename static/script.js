
document.addEventListener('DOMContentLoaded', () => {
    const summaryLevel = document.getElementById('summary-level');
    const levelValue = document.getElementById('level-value');
    const modelType = document.getElementById('model-type');
    const customModel = document.getElementById('custom-model');
    const uploadBtn = document.getElementById('upload-model');
    const summarizeBtn = document.getElementById('summarize-btn');

    summaryLevel.addEventListener('input', (e) => {
        levelValue.textContent = `${e.target.value}%`;
    });

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
        const level = summaryLevel.value;
        const model = modelType.value;

        if (!text) {
            alert('Please enter text to summarize');
            return;
        }

        try {
            const response = await fetch('/process_text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text, level, model })
            });

            const result = await response.json();
            
            document.getElementById('summary-output').textContent = result.summary;
            document.getElementById('compression-ratio').textContent = 
                `Compression: ${result.compression_ratio}%`;
            document.getElementById('processing-time').textContent = 
                `Processing Time: ${result.processing_time}ms`;
            
            if (result.wordcloud_path) {
                const wordcloudImg = document.getElementById('wordcloud-img');
                wordcloudImg.src = result.wordcloud_path;
                wordcloudImg.style.display = 'block';
            }
        } catch (error) {
            alert('Error processing text');
        }
    });
});
