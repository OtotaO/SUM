
document.addEventListener('DOMContentLoaded', () => {
    const summaryLevel = document.getElementById('summary-level');
    const summaryValue = document.getElementById('summary-value');
    const modelSelect = document.getElementById('model-select');
    const uploadBtn = document.getElementById('upload-model');
    const customModel = document.getElementById('custom-model');
    
    summaryLevel.addEventListener('input', () => {
        summaryValue.textContent = `${summaryLevel.value}%`;
    });
    
    uploadBtn.addEventListener('click', () => {
        customModel.click();
    });
    
    document.getElementById('summarize-btn').addEventListener('click', async () => {
        const text = document.getElementById('input-text').value;
        const level = summaryLevel.value;
        const model = modelSelect.value;
        
        try {
            const response = await fetch('/summarize', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    text,
                    level,
                    model
                })
            });
            
            const result = await response.json();
            document.getElementById('summary-output').textContent = result.summary;
            document.getElementById('compression-ratio').textContent = 
                `Compression: ${result.compression_ratio}%`;
            document.getElementById('processing-time').textContent = 
                `Time: ${result.processing_time}ms`;
        } catch (error) {
            console.error('Error:', error);
        }
    });
    
    customModel.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        const formData = new FormData();
        formData.append('model', file);
        
        try {
            await fetch('/upload_model', {
                method: 'POST',
                body: formData
            });
            modelSelect.value = 'custom';
        } catch (error) {
            console.error('Error uploading model:', error);
        }
    });
});
