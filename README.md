# Threat Detection Research Models

This repository contains a comprehensive collection of models for real-time threat detection through explainable multimodal analysis of facial and vocal cues. The repository implements 12 state-of-the-art models from contemporary research, spanning diverse architectures and approaches.

## Project Overview

This research investigates multimodal approaches for real-time threat detection through facial and vocal cues, with a core focus on model explainability. Using benchmark datasets including the Real-Life Violence Situations Dataset, RAVDESS emotional speech corpus, and specialized facial expression collections, we apply multiple explainability methods (LIME, SHAP, Grad-CAM, DeepDream, and feature visualization) to provide interpretable insights into model decision-making processes.

## Models Included

- **Model01_Multimodal_Adaboost**: Ensemble learning approach combining multiple modalities for threat detection
- **Model02_Deep_Learning_Surveillance**: Deep learning approach for suspicious activity detection from surveillance footage
- **Model03_Violence_Detection**: Real-time violence detection in surveillance videos
- **Model04_Multimodal_Emotion_Recognition**: Real-time multimodal human emotion recognition with deep learning
- **Model05_FacialCueNet**: Interpretable model for criminal interrogation using facial expressions
- **Model06_Victim_Tracking_LSTM**: System for recognizing signs of violence using Long-Short Term Memory networks
- **Model07_Face_Analysis_System**: Face detection, quality assessment, and biometric matching system
- **Model08_CNN_Violence_Detection**: CNN-based approach for violence detection
- **Model09_Face_Embedding**: Facial feature extraction and embedding analysis

Each model is provided as both a Jupyter Notebook (`.ipynb`) for interactive exploration and as a Python script (`.py`) for deployment.

## Key Findings

- Multimodal approaches achieve superior accuracy (88-94%) compared to unimodal techniques
- Attention-based fusion mechanisms prove particularly effective at dynamically weighting input modalities
- Explainability analysis revealed that temporal dynamics in facial micro-expressions and spectral patterns in vocal features were the most influential threat indicators
- Eye region movements (23%) and mouth tension (18%) contribute significantly to predictions
- Critical limitations in current approaches include overreliance on computationally expensive architectures and limited exploration of lightweight alternatives

## Repository Structure

```
├── Model01_Multimodal_Adaboost.[py|ipynb]   # Multimodal Adaboost implementation
├── Model02_Deep_Learning_Surveillance.[py|ipynb]  # Surveillance activity detection
├── Model03_Violence_Detection.[py|ipynb]  # Real-time violence detection
├── Model04_Multimodal_Emotion_Recognition.[py|ipynb]  # Emotion recognition
├── Model05_FacialCueNet.[py|ipynb]  # FacialCueNet implementation
├── Model06_Victim_Tracking_LSTM.[py|ipynb]  # Victim tracking system
├── Model07_Face_Analysis_System.[py|ipynb]  # Face analysis and biometric matching
├── Model08_CNN_Violence_Detection.[py|ipynb]  # CNN-based violence detection
└── Model09_Face_Embedding.[py|ipynb]  # Facial feature embedding analysis
```

## Dataset Requirements

The models in this repository were trained and evaluated using the following datasets:

1. **Real-Life Violence Situations Dataset**: Contains 1,000 videos equally divided between violent and non-violent situations
2. **FER-2013 Dataset**: Contains 35,887 grayscale images of facial expressions categorized into seven emotions
3. **RAVDESS Emotional Speech Audio Dataset**: Contains 1,440 audio recordings from 24 professional actors vocalizing with different emotions

## Usage

Each model can be run independently through either:

1. **Jupyter Notebooks**: Open the `.ipynb` files in Jupyter Lab/Notebook for an interactive experience with visualizations
2. **Python Scripts**: Run the `.py` files for deployment or integration into larger systems

### Prerequisites

```
pip install -r requirements.txt
```

Required packages include:

- TensorFlow/PyTorch
- OpenCV
- Librosa
- scikit-learn
- LIME, SHAP
- matplotlib, seaborn
- face_recognition

## Explainability Techniques

The repository implements various explainability techniques to understand model decision-making:

- **LIME**: Local Interpretable Model-agnostic Explanations
- **SHAP**: SHapley Additive exPlanations
- **Grad-CAM**: Gradient-weighted Class Activation Mapping
- **DeepDream**: Visualization of patterns learned by the models
- **Feature Importance Analysis**: Systematic perturbation to understand feature impact

## References

1. Patil et al. (2023). "Real-Time Violence Detection in Surveillance Videos"
2. Amrutha et al. (2020). "Deep Learning Approach for Suspicious Activity Detection from Surveillance Footage"
3. Poria et al. (2016). "Fusing Audio, Visual, and Textual Clues for Sentiment Analysis from Multimodal Content"
4. Hosseini and Yamaghani (2024). "Real Time Multimodal Human Emotion Recognition: A Deep Learning Approach"
5. Nam et al. (2023). "FacialCueNet: Unmasking Deception- An Interpretable Model for Criminal Interrogation Using Facial Expressions"
6. Gheni et al. (2024). "A Victim Tracking System by Recognizing Signs of Violence Using Long-Short Term Memory"
7. Westlake et al. (2022). "Developing Automated Methods to Detect and Match Face and Voice Biometrics in Child Sexual Abuse Videos"
8. Bermejo Nievas et al. (2016). "Multi-modal Human Aggression Detection"
9. Alghowinem et al. (2023). "Multimodal Region-Based Behavioral Modeling for Suicide Risk Screening"

## License

This project is available for academic and research purposes. Please cite the repository if you use any part of this work.

## Contributors

- Mostafa Nashaat
- Fares Wael
- Abdelwahab Hassan
- Ahmed Abdelsamad
