from django.shortcuts import render, redirect
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
from torch.autograd import Variable
import time
import sys
from torch import nn
import json
import glob
import copy
from torchvision import models
import shutil
from PIL import Image as pImage
import time
from django.conf import settings
from .forms import VideoUploadForm

index_template_name = 'index.html'
predict_template_name = 'predict.html'
about_template_name = "about.html"

im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
sm = nn.Softmax()
inv_normalize = transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size,im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:,-1,:]))

class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100/self.count)
        first_frame = np.random.randint(0, a)
        
        for i, frame in enumerate(self.frame_extract(video_path)):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                pass
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
                
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)
    
    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path) 
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image

def im_convert(tensor, video_file_name):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image

def predict(model, img, path='./', video_file_name=""):
    fmap, logits = model(img.to(device))
    img = im_convert(img[:,-1,:,:,:], video_file_name)
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:,int(prediction.item())].item() * 100
    print('confidence of prediction:', confidence)  
    return [int(prediction.item()), confidence]

def get_accurate_model(sequence_length):
    models_dir = os.path.join(settings.PROJECT_DIR, "models")
    model_files = glob.glob(os.path.join(models_dir, "*.pt"))
    
    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")
    
    # Find model with matching sequence length
    for model_path in model_files:
        try:
            filename = os.path.basename(model_path)
            parts = filename.split('_')
            if len(parts) >= 4 and int(parts[3]) == sequence_length:
                return model_path
        except (IndexError, ValueError):
            continue
    
    # If no exact match, return first model
    return model_files[0]

ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'gif', 'webm', 'avi', '3gp', 'wmv', 'flv', 'mkv'}

def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

def index(request):
    if request.method == 'GET':
        video_upload_form = VideoUploadForm()
        if 'file_name' in request.session:
            del request.session['file_name']
        if 'preprocessed_images' in request.session:
            del request.session['preprocessed_images']
        if 'faces_cropped_images' in request.session:
            del request.session['faces_cropped_images']
        return render(request, index_template_name, {"form": video_upload_form})
    else:
        video_upload_form = VideoUploadForm(request.POST, request.FILES)
        if video_upload_form.is_valid():
            video_file = video_upload_form.cleaned_data['upload_video_file']
            video_file_ext = video_file.name.split('.')[-1]
            sequence_length = video_upload_form.cleaned_data['sequence_length']
            
            if not allowed_video_file(video_file.name):
                video_upload_form.add_error("upload_video_file", "Only video files are allowed")
                return render(request, index_template_name, {"form": video_upload_form})
            
            if sequence_length <= 0:
                video_upload_form.add_error("sequence_length", "Sequence Length must be greater than 0")
                return render(request, index_template_name, {"form": video_upload_form})
            
            saved_video_file = f'uploaded_file_{int(time.time())}.{video_file_ext}'
            upload_path = os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file)
            
            with open(upload_path, 'wb') as vFile:
                shutil.copyfileobj(video_file, vFile)
            
            request.session['file_name'] = upload_path
            request.session['sequence_length'] = sequence_length
            return redirect('ml_app:predict')
        else:
            return render(request, index_template_name, {"form": video_upload_form})

def predict_page(request):
    if request.method == "GET":
        if 'file_name' not in request.session:
            return redirect("ml_app:home")
            
        video_file = request.session['file_name']
        sequence_length = request.session['sequence_length']
        video_file_name = os.path.basename(video_file)
        video_file_name_only = os.path.splitext(video_file_name)[0]
        
        # Load model
        model_path = get_accurate_model(sequence_length)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = Model(2).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Process video
        video_dataset = validation_dataset([video_file], sequence_length=sequence_length, transform=train_transforms)
        
        # Extract frames
        cap = cv2.VideoCapture(video_file)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        cap.release()
        
        # Process frames
        preprocessed_images = []
        faces_cropped_images = []
        padding = 40
        faces_found = 0
        
        for i in range(min(sequence_length, len(frames))):
            frame = frames[i]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save preprocessed image
            image_name = f"{video_file_name_only}_preprocessed_{i+1}.png"
            image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
            pImage.fromarray(rgb_frame, 'RGB').save(image_path)
            preprocessed_images.append(image_name)
            
            # Face detection and cropping
            face_locations = face_recognition.face_locations(rgb_frame)
            if face_locations:
                top, right, bottom, left = face_locations[0]
                frame_face = frame[
                    max(0, top - padding):min(frame.shape[0], bottom + padding),
                    max(0, left - padding):min(frame.shape[1], right + padding)
                ]
                rgb_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
                image_name = f"{video_file_name_only}_cropped_faces_{i+1}.png"
                image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
                pImage.fromarray(rgb_face, 'RGB').save(image_path)
                faces_found += 1
                faces_cropped_images.append(image_name)
        
        if faces_found == 0:
            return render(request, predict_template_name, {"no_faces": True})
        
        # Make prediction
        try:
            prediction = predict(model, video_dataset[0], './', video_file_name_only)
            confidence = round(prediction[1], 1)
            output = "REAL" if prediction[0] == 1 else "FAKE"
            
            context = {
                'preprocessed_images': preprocessed_images,
                'faces_cropped_images': faces_cropped_images,
                'original_video': video_file_name,
                'output': output,
                'confidence': confidence
            }
            
            return render(request, predict_template_name, context)
            
        except Exception as e:
            print(f"Exception occurred during prediction: {e}")
            return render(request, 'cuda_full.html')

def about(request):
    return render(request, about_template_name)

def handler404(request, exception):
    return render(request, '404.html', status=404)

def cuda_full(request):
    return render(request, 'cuda_full.html')