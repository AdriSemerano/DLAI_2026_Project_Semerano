# DLAI_2026_Project_Semerano
This project addresses the Sound Event Detection (SED) task for the Kaggle BirdCLEF+ 2026 competition. Two custom CNN architectures based on Mel spectrograms were designed and trained from scratch. Their effectiveness is evaluated both individually and combined in an ensemble, as well as compared against an ensemble of pre-trained models.

The pre-trained baselines were initialized using the timm library and fine-tuned on the challenge dataset across both Phase A and Phase B. 
The resulting model weights of both, Pre-Trained and From-Scratch models, are available for download via a Google Drive link provided in the trained_models_weights directory.
