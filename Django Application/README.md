

## 🖥️ Running the Application **Locally**

### Step 1: Clone the Repository

```bash
git clone https://github.com/abhijitjadhav1998/Deepfake_detection_using_deep_learning.git
cd Deepfake_detection_using_deep_learning/Django\ Application/
```

### Step 2: (Optional) Create and Activate Virtual Environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare Model Files

* Download the trained model from this [Google Drive link](https://drive.google.com/drive/folders/1UX8jXUXyEjhLLZ38tcgOwGsZ6XFSLDJ-?usp=sharing)
* Create the necessary directories in your project root if not already there:

  ```bash
  mkdir models uploaded_images uploaded_videos
  ```
* Copy the model file (e.g., `model_84_acc_10_frames_final_data.pt`) to the `models/` folder.

### Step 5: Run the Application

```bash
python manage.py runserver
```

---

## 🐳 Running the Application Using **Docker**

### Step 1: Ensure Docker and Nvidia GPU support are installed and running

### Step 2: Run the Deepfake Detection Container

```bash
docker run --rm --gpus all \
  -v static_volume:/home/app/staticfiles/ \
  -v media_volume:/app/uploaded_videos/ \
  --name=deepfakeapplication \
  abhijitjadhav1998/deefake-detection-20framemodel
```

### Step 3: Run the Nginx Reverse Proxy Server

```bash
docker run -p 80:80 --volumes-from deepfakeapplication \
  -v static_volume:/home/app/staticfiles/ \
  -v media_volume:/app/uploaded_videos/ \
  abhijitjadhav1998/deepfake-nginx-proxyserver
```

### Step 4: Access the Application

* Open your browser and go to: [http://localhost:80](http://localhost:80)

---

Let me know if you want a simplified version or a bash script for setup!
