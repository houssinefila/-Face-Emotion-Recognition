# Reconnaissance des émotions faciales en temps réel

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13-orange)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-green)

## 📄 Description

Ce projet utilise la **caméra de votre ordinateur** pour détecter les **émotions du visage en temps réel**.  
Il se base sur le modèle pré-entraîné **[abhilash88/face-emotion-detection](https://huggingface.co/abhilash88/face-emotion-detection)** de Hugging Face, entraîné sur le dataset FER2013.

Le programme affiche **l’émotion détectée** et **le pourcentage de confiance** directement sur le flux vidéo.

---

## 🔧 Prérequis

- Python 3.8 ou plus récent  
- Bibliothèques Python nécessaires :
```bash
pip install torch transformers pillow opencv-python
