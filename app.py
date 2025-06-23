from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import tempfile
import os
import uuid
import numpy as np
import soundfile as sf
import scipy.signal as signal

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}
TEMP_DIR = tempfile.gettempdir()


# Helpers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_to_wav(filepath):
    audio = AudioSegment.from_file(filepath)
    new_path = os.path.splitext(filepath)[0] + '.wav'
    audio.export(new_path, format="wav")
    return new_path


def denoise_with_stft(input_path):
    y, sr = sf.read(input_path)

    # Convert stereo to mono if needed
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Normalize input
    y = y / np.max(np.abs(y))

    # STFT parameters
    n_fft = 2048
    hop_length = 512
    f, t, Zxx = signal.stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

    # Estimate noise profile from first 0.5 seconds
    noise_frames = int((0.5 * sr - n_fft) // hop_length)
    noise_profile = np.mean(np.abs(Zxx[:, :noise_frames]), axis=1, keepdims=True)

    # Spectral gating
    alpha = 1.5  # Adjust aggressiveness
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)
    magnitude_denoised = np.maximum(magnitude - alpha * noise_profile, 0)
    Zxx_denoised = magnitude_denoised * np.exp(1j * phase)

    # Inverse STFT
    _, y_denoised = signal.istft(Zxx_denoised, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)

    # Normalize output
    y_denoised = y_denoised / np.max(np.abs(y_denoised))

    # Save denoised audio
    cleaned_path = input_path.replace(".wav", "_cleaned.wav")
    sf.write(cleaned_path, y_denoised, sr)

    return cleaned_path


def trim_audio_by_parts(path, num_parts):
    audio = AudioSegment.from_wav(path)
    duration = len(audio)
    part_len = duration // num_parts
    parts = []
    for i in range(num_parts):
        start = i * part_len
        end = duration if i == num_parts - 1 else (i + 1) * part_len
        parts.append(audio[start:end])
    return parts


def trim_audio_by_range(path, start_sec, end_sec):
    audio = AudioSegment.from_wav(path)
    return audio[start_sec * 1000:end_sec * 1000]


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('audio_file')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        base_uuid = str(uuid.uuid4())
        temp_path = os.path.join(TEMP_DIR, base_uuid + "_" + filename)
        file.save(temp_path)

        # Convert to WAV if needed
        if not filename.lower().endswith('.wav'):
            temp_path = convert_to_wav(temp_path)

        # Store original path
        original_wav = temp_path

        # Apply denoising
        denoised_wav = denoise_with_stft(temp_path)

        # Pass both filenames to template
        return redirect(url_for('preprocess', original=os.path.basename(original_wav),
                                denoised=os.path.basename(denoised_wav)))
    return 'Invalid file format', 400



@app.route('/preprocess')
def preprocess():
    original = request.args.get('original')
    denoised = request.args.get('denoised')
    return render_template('preprocess.html', original=original, denoised=denoised)


@app.route('/process', methods=['POST'])
def process():
    filename = request.form['filename']
    input_path = os.path.join(TEMP_DIR, filename)

    processed_files = []
    num_parts = request.form.get('num_parts')
    start = request.form.get('start_time')
    end = request.form.get('end_time')

    if num_parts:
        parts = trim_audio_by_parts(input_path, int(num_parts))
        for part in parts:
            part_name = f"{uuid.uuid4()}.wav"
            part_path = os.path.join(TEMP_DIR, part_name)
            part.export(part_path, format='wav')
            processed_files.append(part_name)
    elif start and end:
        trimmed = trim_audio_by_range(input_path, float(start), float(end))
        part_name = f"{uuid.uuid4()}.wav"
        part_path = os.path.join(TEMP_DIR, part_name)
        trimmed.export(part_path, format='wav')
        processed_files.append(part_name)

    return render_template('download.html', files=processed_files)


@app.route('/temp/<filename>')
def serve_temp_file(filename):
    path = os.path.join(TEMP_DIR, filename)
    return send_file(path, as_attachment=False)


@app.route('/download/<filename>')
def download_temp_file(filename):
    path = os.path.join(TEMP_DIR, filename)
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)