import base64, json, re, urllib.error, urllib.request
from flask import Flask, jsonify, render_template, request
from math_router import solve_math
app = Flask(__name__)
OLLAMA_URL = 'http://127.0.0.1:11434/api/chat'
TEXT_MODEL = 'qwen2.5:7b'
VISION_MODEL = 'qwen2.5vl:3b'
AI_TIMEOUT = 300
MAX_IMAGE_SIZE = 10 * 1024 * 1024
SYSTEM_PROMPT = '''You are Math Solver AI, a friendly, careful mathematics teacher. Use clear, simple English and short sentences. Teach in small steps and explain WHY each important step works. Answer the exact question. Use previous messages for follow-up questions. If the student says “solve part b” and an image is attached, inspect that image and solve the requested part. Never invent unreadable labels or measurements; ask for clarification if needed. For equations, show each operation on both sides. For fractions, explain common denominators. For geometry, state the formula, explain symbols, substitute values, calculate, and include units. Use numbered Markdown steps and readable LaTeX equations. End with **FINAL ANSWER:**. Do not claim to verify anything unless you did.'''
VISION_PROMPT = '''The student has attached a mathematics question image. Carefully read all visible text, numbers, signs, fractions, powers, question numbers, subparts, diagram labels, and measurements. Follow the student's exact instruction, for example question 9(b). Do not invent information. Solve in clear simple English with every important step.'''

def decode_image(value):
    if not value: return None
    if not isinstance(value, str): raise ValueError('The image data format is invalid.')
    m = re.match(r'^data:image/(jpeg|jpg|png|webp);base64,([A-Za-z0-9+/=\s]+)$', value, re.I)
    if not m: raise ValueError('Please upload a JPG, PNG, or WebP image.')
    try: raw = base64.b64decode(re.sub(r'\s+', '', m.group(2)), validate=True)
    except Exception as e: raise ValueError('The image could not be read. Please upload it again.') from e
    if not raw: raise ValueError('The selected image is empty.')
    if len(raw) > MAX_IMAGE_SIZE: raise ValueError('Please use an image smaller than 10 MB.')
    return base64.b64encode(raw).decode('ascii')

def ask_ollama(question, history=None, image_b64=None):
    messages = [{'role':'system','content':SYSTEM_PROMPT}]
    if image_b64: messages.append({'role':'user','content':VISION_PROMPT,'images':[image_b64]})
    if isinstance(history, list):
        for item in history[-10:]:
            if isinstance(item, dict) and item.get('role') in ('user','assistant') and isinstance(item.get('content'), str) and item['content'].strip():
                messages.append({'role':item['role'],'content':item['content'][:8000]})
    last = {'role':'user','content':question}
    if image_b64: last['images'] = [image_b64]
    messages.append(last)
    payload = {'model': VISION_MODEL if image_b64 else TEXT_MODEL, 'messages':messages, 'stream':False, 'options':{'temperature':0.1}}
    req = urllib.request.Request(OLLAMA_URL, data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=AI_TIMEOUT) as r: data = json.loads(r.read().decode())
        answer = data.get('message',{}).get('content','').strip()
        if not answer: raise RuntimeError('The AI returned an empty answer.')
        return answer
    except urllib.error.HTTPError as e:
        if e.code == 404: raise RuntimeError(f'Ollama could not find a model. Run `ollama list`. Text: {TEXT_MODEL}; vision: {VISION_MODEL}.')
        raise RuntimeError(f'Ollama returned HTTP error {e.code}.')
    except urllib.error.URLError as e: raise RuntimeError('I could not connect to Ollama. Make sure Ollama is running on your Mac.') from e
    except TimeoutError as e: raise RuntimeError('The AI took too long to respond. Please try again.') from e

@app.route('/')
def home(): return render_template('index.html')
@app.route('/solve', methods=['POST'])
def solve():
    data = request.get_json(silent=True) or {}
    question = str(data.get('question') or data.get('message') or '').strip()
    try: image_b64 = decode_image(data.get('image'))
    except ValueError as e: return jsonify({'error':str(e)}), 400
    if not question and not image_b64: return jsonify({'error':'Please type a question or upload a photo.'}), 400
    if not question: question = 'Read this mathematics question and explain the solution step by step.'
    try: return jsonify({'answer':solve_math(question, image_b64, data.get('history', []))})
    except Exception as e:
        app.logger.exception('Solve request failed')
        return jsonify({'error':str(e) or 'Something went wrong. Please try again.'}), 503
@app.route('/reset', methods=['POST'])
def reset(): return jsonify({'status':'ok'})
@app.route('/health')
def health(): return jsonify({'app':'Math Solver AI','status':'running','text_model':TEXT_MODEL,'vision_model':VISION_MODEL})
if __name__ == '__main__': app.run(host='127.0.0.1', port=5000, debug=True)
