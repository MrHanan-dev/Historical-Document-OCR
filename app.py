from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
from pathlib import Path
from datetime import datetime
from main import process_directory, run_entity_extraction
from gemini_client import GeminiClient
import prompts

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flash messages

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = Path('input_images')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

UPLOAD_FOLDER.mkdir(exist_ok=True)
Path('output/transcriptions').mkdir(parents=True, exist_ok=True)
Path('output/final_outputs').mkdir(parents=True, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Get list of processed files
    transcriptions_dir = Path('output/transcriptions')
    transcriptions = []
    if transcriptions_dir.exists():
        transcriptions = [f for f in os.listdir(transcriptions_dir) if f.endswith('.txt')]
    
    return render_template('index.html', transcriptions=transcriptions)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Clear the input_images directory
            for f in UPLOAD_FOLDER.glob('*'):
                f.unlink()
            
            # Save the new file
            filepath = UPLOAD_FOLDER / file.filename
            file.save(filepath)
            
            # Initialize Gemini client
            client = GeminiClient()
            
            # Process the image with the full transcription prompt
            process_directory(
                input_dir=UPLOAD_FOLDER,
                output_dir=Path('output'),
                client=client,
                prompt=prompts.TRANSCRIPTION_PROMPT
            )
            
            # Run entity extraction with the proper prompt
            run_entity_extraction(
                input_dir=Path('output/transcriptions'),
                output_dir=Path('output/final_outputs'),
                client=client,
                prompt_template=prompts.ENTITY_EXTRACTION_PROMPT
            )
            
            flash('Document successfully processed! You can view the results below.', 'success')
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            
        return redirect(url_for('index'))
    
    flash('Invalid file type')
    return redirect(request.url)

@app.route('/transcription/<filename>')
def view_transcription(filename):
    try:
        with open(Path('output/transcriptions') / filename, 'r', encoding='utf-8') as f:
            content = f.read()
        return render_template('transcription.html', 
                            content=content, 
                            filename=filename, 
                            datetime=datetime)
    except:
        flash('Error reading transcription', 'error')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
