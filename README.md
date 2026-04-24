# 4240 Project
This is a real-time facial recognition system written in MATLAB. 

### Add-ons Required:
- MATLAB Support Package for USB Webcams
- Computer Vision Toolbox

### What It Is
A live, webcam-based security camera system with a custom graphical user interface. It detects faces in real-time, extracts visual features, and matches them against a dynamic database of registered users, providing a live "Known/Unknown" status and confidence percentage.

### Methods
- **Preprocessing:** Converts live frames to grayscale, resizes them to a fixed 100x100 resolution, and applies adaptive histogram equalization (CLAHE) to handle basic lighting differences.
- **Feature Extraction (Feature Fusion):** Uses a combined approach for higher accuracy:
  - **PCA (Eigenfaces):** Evaluates the holistic, global pixel intensity of the face.
  - **HOG (Histogram of Oriented Gradients):** Evaluates local shapes and edge contours, making the model more robust to shadows.
- **Normalization:** Uses Independent L2 Normalization so both the PCA and HOG arrays are mathematically forced to carry exactly 50% of the weight during comparison. 
- **Classification:** Uses Cosine Similarity to calculate the distance between the live webcam face and the known database vectors.

### Pipeline
Live Frame ➔ Cascade Object Detector ➔ Tightly Crop Face ➔ Preprocess ➔ Extract PCA + HOG ➔ L2 Normalize ➔ Cosine Similarity Match ➔ Output Result to UI

### How to Run
1. Ensure your webcam is connected and the required MATLAB add-ons are installed.
2. Open MATLAB and navigate to the project folder.
3. Open and run `main.m` to launch the GUI.
4. Click **Register New Person** to add a subject to the database (the system will guide you through taking 5 quick poses).
5. Click **Start Camera** to begin the live face tracking and recognition feed.