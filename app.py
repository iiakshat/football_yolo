import os
import json
import time
import threading
from main import process
from flask import Flask, request, redirect, url_for, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'output/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

processing_status = {'status': 'idle'}

def process_video(filename, config):
    config_path = "config.json"
    with open(config_path, 'r') as file:
        configog = json.load(file)

    for k in config.copy():
        if config[k] == "true":
            config[k] = True

        if config[k] == "":
            config.pop(k)
    
    for k in configog:
        if (k.startswith("show") or k=="cache") and config.get(k, False) != True:
                config[k] = False

    config["input_path"] = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    config["alpha"] = float(config["alpha"])
    numericalvals = ["frame_rate", "min_distance_to_assign_ball", "court_width", "court_length"]
    paths = ["model", "movementpath", "trackpath"]
    
    for path in paths:
        if path in config and not os.path.exists(config[path]):
            config.pop(path)

    for val in numericalvals: 
        if val in config:
            try:
                config[val] = int(config[val])
            except:
                config.pop(val)

    configog.update(config)
    process(configog)
    processing_status['status'] = 'done'

def clearfiles():
    time.sleep(5)
    try:
        os.remove(UPLOAD_FOLDER + processing_status["title"])
    except:
        pass

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_video():
	if 'file' not in request.files:
		print('No file uploaded.')
		return redirect(request.url)

	file = request.files['file']
	if file.filename == '':
		print('Invalid file format')
		return redirect(request.url)

	filename = secure_filename(file.filename)
	file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
	processing_status['status'] = 'processing'
	processing_status['title'] = filename
	config = dict(request.form)
	threading.Thread(target=process_video, args=(filename,config,)).start()
	return redirect(url_for('processing'))

@app.route('/processing')
def processing():
    return render_template('process.html', filename="uploads/" + processing_status["title"])

@app.route('/status')
def status():
    return jsonify(processing_status)

@app.route('/final')
def final():
    threading.Thread(target=clearfiles).start()
    return render_template('final.html', filename=processing_status["title"])

@app.route('/display/<filename>')
def display(filename):
    return redirect(url_for('static', filename=filename), code=301)

@app.route('/output/<filename>')
def output_video(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
