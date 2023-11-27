from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

fall_status = False
last_sent_status = None

def event_stream():
    global last_sent_status
    count = 0
    while True:
        status = 1 if fall_status else 0
        if last_sent_status != status:
            yield f"data: {status} at {count}\n\n"
            last_sent_status = status
            count += 1


@app.route('/fall_status_stream')
def sse_fall_status():
    response = Response(event_stream(), content_type='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    return response

@app.route('/fall_status', methods=['POST'])
def update_fall_status():
    global fall_status
    data = request.json
    if 'fall_status' in data:
        fall_status = data['fall_status']
        return jsonify({'message': 'Fall status updated'}), 200
    else:
        return jsonify({'message': 'Invalid data'}), 400

@app.route('/fall_status', methods=['GET'])
def status():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
