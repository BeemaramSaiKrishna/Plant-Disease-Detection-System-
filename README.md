# 🌱 Plant Disease Prediction - CNN Image Classifier

This project builds a **Convolutional Neural Network (CNN)** based image classifier to detect and predict **plant diseases** from leaf images.  
The model aims to assist farmers and researchers by providing an efficient and accurate diagnosis based on visual symptoms.

---

## 📄 Project Overview

- **Model**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow / Keras
- **Input**: Images of plant leaves
- **Output**: Predicted disease category
- **Dataset**: Publicly available plant disease datasets (e.g., PlantVillage) or a custom-curated dataset.

---

## 📚 Features

- Preprocessing and augmentation of plant leaf images
- CNN architecture for feature extraction and classification
- Model training with accuracy and loss monitoring
- Evaluation on validation/test data
- Prediction function for new/unseen images
- Export trained model for deployment

---

## 🚰 Tech Stack

- Python 3.x
- Jupyter Notebook (.ipynb)
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- OpenCV (optional for preprocessing)
- Streamlit (for deployment)
- pillow
 



---

## 🧀 Model Architecture

The CNN typically includes:
- Convolutional layers
- MaxPooling layers
- Dropout layers (to prevent overfitting)
- Fully connected Dense layers
- Softmax activation for final output

*(Customize this section if you used a specific architecture like ResNet, MobileNet, or a custom CNN.)*

---

## 📊 Results

- **Training Accuracy**: ~ (update with your accuracy, e.g., 95%)
- **Validation Accuracy**: ~ (e.g., 93%)

Graphs of:
- Training vs Validation Accuracy
- Training vs Validation Loss

*(Include screenshots in your GitHub repo if possible.)*

---

## 🚀 How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/final_Plant_Disease_Prediction_CNN_Image_Classifier.git
    cd final_Plant_Disease_Prediction_CNN_Image_Classifier
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Open the notebook:
    ```bash
    jupyter notebook final_Plant_Disease_Prediction_CNN_Image_Classifier_(1).ipynb
    ```

4. Run all cells to train the model or load a pre-trained model if provided.

---

## 🚀 How to Deploy Using Streamlit

1. Install Streamlit:
    ```bash
    pip install streamlit
    ```

2. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
   *(Assuming your Streamlit app code is in `app.py`. Update the filename if different.)*

3. Your web app will open automatically in your default browser at:
    ```
    http://localhost:8501
    ```

---

## 📄 About the Streamlit App

- Upload a plant leaf image.
- The model will process the image and predict the disease.
- Simple, fast, and lightweight UI for quick diagnosis.

---

## 👅 Dataset

- (Mention the dataset source, e.g., [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease))
- Dataset Structure:
    ```
    dataset/
      ├── train/
      ├── test/
      └── validation/
    ```

---

## 📈 Future Work

- Deploy the model using Flask / FastAPI / Streamlit
- Improve model performance using Transfer Learning
- Increase dataset size for better generalization
- Mobile app integration for real-time prediction

---

## 🤝 Contribution

Contributions are welcome!  
Feel free to fork the project, open issues, or submit pull requests.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

 

