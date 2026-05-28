import os
import uuid
from flask import Flask, render_template, request, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from inference_service import GenCADService

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'web_uploads'
app.config['OUTPUT_FOLDER'] = 'web_outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Check if checkpoints exist
CHECKPOINTS = [
    "model/ckpt/ae_ckpt_epoch1000.pth",
    "model/ckpt/ccip_sketch_ckpt_epoch300.pth",
    "model/ckpt/sketch_cond_diffusion_ckpt_epoch1000000.pt"
]

missing = [ckpt for ckpt in CHECKPOINTS if not os.path.exists(ckpt)]
if missing:
    print("\n" + "!"*50)
    print("CRITICAL ERROR: Missing checkpoint files!")
    for m in missing:
        print(f" - {m}")
    print("\nPlease download the checkpoints as described in readme.md")
    print("and place them in the 'model/ckpt/' directory.")
    print("!"*50 + "\n")
    # We'll still try to initialize, but it will likely fail.
    # Alternatively, we could exit, but let's let the service try its best.

# Initialize service (loads models into memory)
print("Initializing GenCAD Service...")
try:
    service = GenCADService()
    print("Service ready.")
except Exception as e:
    print(f"Failed to initialize service: {e}")
    service = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if service is None:
        return jsonify({'error': 'Service is not initialized. Check if checkpoints are missing.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        ext = os.path.splitext(filename)[1]
        
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_id + ext)
        output_filename = unique_id + ".stl"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        file.save(input_path)
        
        print(f"Generating STL for {filename}...")
        success = service.generate_stl(input_path, output_path)
        
        if success:
            print(f"Successfully generated {output_filename}")
            return jsonify({'stl_filename': output_filename})
        else:
            return jsonify({'error': '3D generation failed. The input image might be too complex or invalid.'}), 500

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    # Using 0.0.0.0 to make it accessible in the VM
    app.run(host='0.0.0.0', port=5000, debug=False)
