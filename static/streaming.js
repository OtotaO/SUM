/**
 * streaming.js - Enhanced streaming functionality for SUM
 * 
 * Provides real-time progress updates using Server-Sent Events
 * with a clean, reusable API.
 */

class StreamingClient {
    constructor() {
        this.activeStreams = new Map();
    }

    /**
     * Stream a summarization request
     */
    streamSummarize(text, options = {}) {
        const endpoint = '/api/stream/summarize';
        const data = {
            text: text,
            density: options.density || 'all',
            store_memory: options.storeMemory !== false,
            extract_entities: options.extractEntities || false
        };

        return this._createStream(endpoint, data, options.onProgress);
    }

    /**
     * Stream file processing
     */
    streamFile(file, options = {}) {
        const endpoint = '/api/stream/file';
        const formData = new FormData();
        formData.append('file', file);

        return this._createFileStream(endpoint, formData, options.onProgress);
    }

    /**
     * Stream batch processing
     */
    streamBatch(documents, options = {}) {
        const endpoint = '/api/stream/batch';
        const data = { documents: documents };

        return this._createStream(endpoint, data, options.onProgress);
    }

    /**
     * Stream synthesis
     */
    streamSynthesis(memoryIds, options = {}) {
        const endpoint = '/api/stream/synthesis';
        const data = {
            memory_ids: memoryIds,
            synthesis_type: options.synthesisType || 'comprehensive'
        };

        return this._createStream(endpoint, data, options.onProgress);
    }

    /**
     * Create a streaming connection
     */
    _createStream(endpoint, data, onProgress) {
        return new Promise((resolve, reject) => {
            // Convert to fetch with streaming response
            fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            }).then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                const processStream = async () => {
                    try {
                        while (true) {
                            const { done, value } = await reader.read();
                            
                            if (done) break;
                            
                            buffer += decoder.decode(value, { stream: true });
                            const lines = buffer.split('\n\n');
                            buffer = lines.pop() || '';

                            for (const line of lines) {
                                if (line.trim()) {
                                    const event = this._parseSSE(line);
                                    if (event) {
                                        this._handleEvent(event, onProgress, resolve, reject);
                                    }
                                }
                            }
                        }
                    } catch (error) {
                        reject(error);
                    }
                };

                processStream();
            }).catch(reject);
        });
    }

    /**
     * Create a file streaming connection
     */
    _createFileStream(endpoint, formData, onProgress) {
        return new Promise((resolve, reject) => {
            fetch(endpoint, {
                method: 'POST',
                body: formData
            }).then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                const processStream = async () => {
                    try {
                        while (true) {
                            const { done, value } = await reader.read();
                            
                            if (done) break;
                            
                            buffer += decoder.decode(value, { stream: true });
                            const lines = buffer.split('\n\n');
                            buffer = lines.pop() || '';

                            for (const line of lines) {
                                if (line.trim()) {
                                    const event = this._parseSSE(line);
                                    if (event) {
                                        this._handleEvent(event, onProgress, resolve, reject);
                                    }
                                }
                            }
                        }
                    } catch (error) {
                        reject(error);
                    }
                };

                processStream();
            }).catch(reject);
        });
    }

    /**
     * Parse Server-Sent Event
     */
    _parseSSE(text) {
        const lines = text.split('\n');
        const event = {};

        for (const line of lines) {
            if (line.startsWith('event:')) {
                event.type = line.slice(6).trim();
            } else if (line.startsWith('data:')) {
                try {
                    event.data = JSON.parse(line.slice(5).trim());
                } catch (e) {
                    event.data = line.slice(5).trim();
                }
            }
        }

        return event.data ? event : null;
    }

    /**
     * Handle streaming events
     */
    _handleEvent(event, onProgress, resolve, reject) {
        const data = event.data;

        switch (data.type) {
            case 'start':
                if (onProgress) {
                    onProgress({
                        type: 'start',
                        message: data.message,
                        progress: 0
                    });
                }
                break;

            case 'progress':
                if (onProgress) {
                    onProgress({
                        type: 'progress',
                        message: data.message,
                        progress: data.progress,
                        stats: data.stats
                    });
                }
                break;

            case 'document_start':
            case 'document_complete':
                if (onProgress) {
                    onProgress(data);
                }
                break;

            case 'complete':
                if (onProgress) {
                    onProgress({
                        type: 'complete',
                        message: data.message,
                        progress: 100
                    });
                }
                resolve(data.result || data);
                break;

            case 'error':
                const error = new Error(data.message);
                if (onProgress) {
                    onProgress({
                        type: 'error',
                        message: data.message,
                        error: error
                    });
                }
                reject(error);
                break;
        }
    }

    /**
     * Cancel all active streams
     */
    cancelAll() {
        // In practice, you'd track and cancel active fetch operations
        this.activeStreams.clear();
    }
}

// Create global instance
const streamingClient = new StreamingClient();

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = StreamingClient;
}