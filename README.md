# Chest X-Ray Disease Detection App

Welcome to the Chest X-Ray Based Detection App!  
This project uses a deep learning model trained on Chest X-Ray images to detect three conditions:

- COVID-19
- NORMAL (Healthy)
- PNEUMONIA

**Live Demo:** [Click here to try the Web App](https://chest-xray-based-detection.streamlit.app/)

---

## Team Members

- Arup Saha (ID: 213-134-018)
- Nafiza Islam Nowrin (ID: 221-134-007)

## About the Project

This project is built using:

- A custom-trained Convolutional Neural Network (CNN) achieving 96.35% test accuracy.
- Preprocessing with VGG16 style normalization.
- Mixup data augmentation and various regularization techniques to ensure strong generalization.
- Deployed using Streamlit Cloud for real-time user interaction.

The goal is to provide a simple tool where users can upload Chest X-Ray images and quickly get predictions about potential diseases.

---

## How to Use Locally

If you want to run this project locally:

1. Clone this repository

   ```bash
   git clone https://github.com/ItsArupSaha/streamlit.git
   cd streamlit
   ```

2. Install required libraries

   ```bash
   pip install -r requirements.txt
   pip install streamlit
   ```

3. Run the app
   ```bash
   streamlit run app.py
   ```

Then open the URL shown in your terminal (usually http://localhost:8501).

---

## Project Structure

```
streamlit/
├── app.py                  # Main Streamlit app
├── chest_final_model.keras  # Trained CNN model
├── requirements.txt         # Required libraries
├── All_11_trained_Model_codes/
│   ├── 1st_Model.ipynb
│   ├── 2nd_Model.ipynb
│   └── ... (All experiments and training files)
└── README.md                # This file
```

---

## App Features

- Upload Chest X-Ray images (.png, .jpg, .jpeg supported).
- See instant model predictions.
- View the uploaded image.
- Mobile-friendly UI with simple interface.

---

## Model Details

- 4 Convolutional Layers + Batch Normalization
- Dense Layers with Dropout
- SGD Optimizer with Cosine Decay Scheduler
- Data Augmentation (Mixup, Brightness/Zoom/Flip)
- Trained for 25 epochs with Early Stopping
- Final Test Accuracy: 96.35%

---

## Contribution

Pull Requests are welcome! If you find any issues or improvements, feel free to fork this repo and make a Pull Request.

Steps:

1. Fork this repository.
2. Clone your forked repository.
3. Create a new branch.
4. Commit your changes.
5. Push to your fork.
6. Open a Pull Request!

---

## Acknowledgements

This project was completed as part of an academic course under the supervision of my class teacher.  
Special thanks to all open-source contributors and the dataset providers.

---

## Related Links 🔗

- [Streamlit Documentation](https://docs.streamlit.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Chest X-Ray Dataset Source (Kaggle)](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia)

---

## Disclaimer

This app is for educational and experimental purposes only.  
It is **NOT** intended for clinical use without professional medical validation.

---
