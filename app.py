import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import librosa
import timm
import subprocess
import imageio_ffmpeg
from flask import Flask, render_template, request, jsonify
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from hsemotion.facial_emotions import HSEmotionRecognizer

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_PATH = "model.pth"
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SILENCE_THRESHOLD = 0.015

# --- SECURITY & COMPATIBILITY ---
def setup_security_whitelist():
    safe_classes = []
    try:
        safe_classes.append(torch.nn.modules.linear.Identity)
        safe_classes.append(timm.models.efficientnet.EfficientNet)
    except: pass

    layer_names = [
        'Conv2dSame', 'BatchNormAct2d', 'SqueezeExcite', 
        'Swish', 'Sigmoid', 'DropPath', 'GlobalResponseNorm',
        'DepthwiseSeparableConv', 'InvertedResidual'
    ]
    for name in layer_names:
        try:
            cls = getattr(timm.layers, name, None)
            if cls: safe_classes.append(cls)
            if name == 'BatchNormAct2d':
                from timm.layers.norm_act import BatchNormAct2d
                safe_classes.append(BatchNormAct2d)
            if name == 'Conv2dSame':
                from timm.layers.conv2d_same import Conv2dSame
                safe_classes.append(Conv2dSame)
        except: pass

    if safe_classes:
        torch.serialization.add_safe_globals(safe_classes)

setup_security_whitelist()

original_load = torch.load
def safe_load_wrapper(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load_wrapper

# --- MODEL ARCHITECTURE ---
class NeuroSyncFusion(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.audio_backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        
        self.visual_interface = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device='cpu')
        self.video_backbone = self.visual_interface.model
        
        video_dim = 1280
        if hasattr(self.video_backbone, 'classifier'):
            self.video_backbone.classifier = nn.Identity()
        else:
            self.video_backbone.fc = nn.Identity()

        self.audio_proj = nn.Linear(1024, 512)
        self.video_proj = nn.Linear(video_dim, 512)
        self.cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, audio, video):
        with torch.no_grad():
            a_out = self.audio_backbone(audio).last_hidden_state
        a_feats = self.audio_proj(a_out)
        
        b, f, c, h, w = video.shape
        v_in = video.view(b * f, c, h, w)
        with torch.no_grad():
            v_out = self.video_backbone(v_in)
        v_feats = v_out.view(b, f, -1)
        v_feats = self.video_proj(v_feats)
        
        attn_out, _ = self.cross_attn(query=v_feats, key=a_feats, value=a_feats)
        return self.classifier(torch.mean(attn_out, dim=1))

# --- INITIALIZATION ---
print(f"Loading NeuroSync on {DEVICE}...")
model = NeuroSyncFusion(num_classes=7)

try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'): new_state_dict[k[7:]] = v
        else: new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    print("✅ Model Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- PREPROCESSING ---
def process_data(video_path):
    # 1. Extract Audio
    audio_path = "temp.wav"
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        subprocess.run([ffmpeg_exe, "-y", "-i", video_path, "-vn", "-ar", "16000", "-ac", "1", audio_path], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except:
        return None, None, True

    # 2. Audio Tensor
    try:
        audio, _ = librosa.load(audio_path, sr=16000, duration=7.0)
        rms = np.sqrt(np.mean(audio**2))
        is_silent = bool(rms < SILENCE_THRESHOLD)
        
        audio_input = processor(audio, sampling_rate=16000, max_length=112000, padding="max_length", truncation=True, return_tensors="pt")
        audio_tensor = audio_input.input_values.to(DEVICE)
        
        if is_silent: audio_tensor = torch.zeros_like(audio_tensor)
    except:
        return None, None, True

    # 3. Video Tensor
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    while True:
        ret, frame = cap.read()
        if not ret: break
        count += 1
        if count % 10 == 0: 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
                frame = frame[y:y+h, x:x+w]
            
            img = cv2.resize(frame, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = (img - mean) / std 
            img = np.transpose(img, (2, 0, 1))
            frames.append(img)
    cap.release()

    if len(frames) == 0: 
        video_tensor = torch.zeros((1, 10, 3, 224, 224)).to(DEVICE)
    else:
        video_tensor = torch.tensor(np.array(frames), dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
    return audio_tensor, video_tensor, is_silent

# --- ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files: return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['video']
    file.save("temp.webm")
    
    try:
        a_in, v_in, is_silent = process_data("temp.webm")
        if a_in is None: return jsonify({'error': 'Processing failed'}), 500

        with torch.no_grad():
            logits = model(a_in, v_in)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)
            
        emotion = EMOTIONS[idx.item()]
        
        # Silence Override Logic
        if is_silent and emotion in ['angry', 'fear']:
             emotion = "neutral (Silence)"

        return jsonify({
            'emotion': emotion,
            'confidence': f"{conf.item()*100:.2f}%",
            'scores': {EMOTIONS[i]: f"{probs[i].item():.4f}" for i in range(len(EMOTIONS))},
            'silent': is_silent
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)