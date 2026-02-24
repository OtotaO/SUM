from flask import Flask, request, jsonify
import redis
import hashlib
import time

app = Flask(__name__)
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Dummy summarizer for testing
def summarizer(text, **kwargs):
    return [{'summary_text': 'This is a test summary.'}]

def check_rate_limit(ip):
    key = f"ratelimit:{ip}"
    count = r.incr(key)
    if count == 1:
        r.expire(key, 60)
    return count <= 60

def get_summary_from_cache(text_hash):
    return r.get(f"summary:{text_hash}")

def save_summary_to_cache(text_hash, summary):
    r.setex(f"summary:{text_hash}", 3600, summary)

@app.route('/health')
def health():
    try:
        r.ping()
        redis_ok = True
    except:
        redis_ok = False
    return jsonify({
        'status': 'healthy' if redis_ok else 'degraded',
        'model_loaded': True,
        'redis_connected': redis_ok
    })

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text'}), 400

    text = data['text']
    if not text:
        return jsonify({'error': 'Empty text'}), 400

    if len(text) > 100000:
        return jsonify({'error': 'Text too long (max 100000)'}), 400

    ip = request.remote_addr
    if not check_rate_limit(ip):
        return jsonify({'error': 'Rate limit exceeded'}), 429

    text_hash = hashlib.md5(text.encode()).hexdigest()
    cached = get_summary_from_cache(text_hash)
    if cached:
        return jsonify({'summary': cached, 'cached': True})

    try:
        result = summarizer(text)
        summary = result[0]['summary_text']
        save_summary_to_cache(text_hash, summary)
        return jsonify({'summary': summary, 'cached': False})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def stats():
    return jsonify({
        'cache_keys': r.dbsize(),
        'memory_used_mb': float(r.info().get('used_memory', 0)) / 1024 / 1024,
        'uptime_seconds': int(r.info().get('uptime_in_seconds', 0))
    })

if __name__ == '__main__':
    app.run(port=3000)
