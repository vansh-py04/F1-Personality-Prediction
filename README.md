# F1 Driver Personality Predictor
This project predicts a user's MBTI personality type based on a text input and matches it with an F1 driver who shares the same personality type.

It uses a BERT-based model for accurate natural language understanding and is wrapped in an interactive Dash web app UI for a smooth and stylish experience.

---
# About the Project
Model: Fine-tuned using BERT (bert-base-uncased) to classify MBTI types from text inputs.

Dataset: A large corpus of personality-labeled text was used for training (not included here).

Training: The model was trained on Google Colab for convenience and GPU acceleration.

Frontend: Built using Plotly Dash, allowing real-time predictions and a visual UI with images of real F1 drivers.

---
# What the Model Does
Accepts a sentence or paragraph written by a user.

Processes it through a fine-tuned BERT model.

Predicts one of 16 MBTI personality types.

Matches that personality to a real-life F1 driver.

Displays the prediction and the corresponding driver's name + image.
