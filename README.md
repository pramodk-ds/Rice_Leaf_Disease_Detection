# 🌾 Rice Leaf Disease Detection  

A deep learning project to **detect rice leaf diseases** using **Convolutional Neural Networks (CNNs)**.  
This project was fully developed and executed on **Google Colab** for faster training with GPU support.  

---

## 🚀 Project Overview  
Rice plants are highly susceptible to various leaf diseases that can reduce yield.  
This project leverages **image classification** with **TensorFlow/Keras CNNs** to detect common rice diseases from leaf images.  

---

## 📂 Dataset  
- **Classes**:  
  - Bacterial Blight  
  - Brown Spot  
  - Leaf Smut  
- **Data Source**: Custom rice leaf dataset (images organized in folders by disease).  
- **Preprocessing**:  
  - Image resizing & normalization  
  - Data augmentation (rotation, flipping, zooming) for better generalization  

---

## ⚙️ Tech Stack  
- **Language**: Python 🐍 (Google Colab)  
- **Libraries**:  
  - `numpy`, `pandas` → Data handling  
  - `opencv`, `PIL` → Image processing  
  - `tensorflow/keras` → Deep learning CNN model  
  - `matplotlib`, `seaborn` → Visualization  
  - `scikit-learn` → Model evaluation metrics  
  - `gradio` → Interactive web app interface  

---

## 🔑 Key Steps  
1. **Image Preprocessing**  
   - Resize all images to uniform size  
   - Normalize pixel values (0–1)  
   - Augment data to reduce overfitting  
2. **Model Building**  
   - Convolutional Neural Network (CNN) with Conv2D, MaxPooling, Dropout, Dense layers  
3. **Training & Validation**  
   - Used categorical cross-entropy loss  
   - Optimizer: Adam  
   - EarlyStopping callback for preventing overfitting  
4. **Evaluation**  
   - Confusion Matrix, Accuracy, Precision, Recall, F1-score  
5. **Deployment**  
   - Integrated with **Gradio** for an easy-to-use image upload and prediction interface  

---

## 🤖 Model Performance  

| Metric              | Score   |
|---------------------|---------|
| Training Accuracy   | ~95%    |
| Validation Accuracy | ~90%    |
| Test Accuracy       | **87%** |

✅ The model achieved **87% test accuracy** on unseen rice leaf images.  

---

## ⚔️ Challenges Faced  
- **Data Duplicates**: Some images were repeated across folders, causing data leakage. Fixed by cleaning dataset.  
- **Overfitting**: Initially model memorized training data. Solved with data augmentation & dropout layers.  
- **Imbalanced Classes**: Some diseases had fewer images; addressed by augmenting minority class samples.  
- **Deployment Issues**: Resolved errors in **Gradio app** integration for prediction UI.  

---

## ▶️ How to Run  
```bash
# Clone this repo
git clone https://github.com/yourusername/Rice-Leaf-Disease-Detection.git
cd Rice-Leaf-Disease-Detection

# Install dependencies
pip install -r requirements.txt

# Run Google Colab notebook
Upload RiceLeaf_Disease_Detection.ipynb to Google Colab and execute step by step.

# Launch Gradio app (browser will open automatically)
python -m webbrowser
```

---

## 🌟 Future Improvements  
- Experiment with **Transfer Learning (ResNet, EfficientNet)** for higher accuracy  
- Expand dataset with more rice diseases  
- Deploy on cloud platforms (Streamlit/Flask + HuggingFace Spaces)  

---

## 👨‍💻 Author
**Pramod K**  
Data Science Enthusiast | Machine Learning | Deep Learning

---
