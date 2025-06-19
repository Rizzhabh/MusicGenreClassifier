This deep learning project aims to classify music genres using a Convolutional Neural Network (CNN). The core idea is to convert music clips into spectrogram images (visual representations of sound) and train a CNN to recognize patterns associated with different genres like classical, pop, jazz, etc.

📁 Dataset
We use the popular GTZAN Genre Collection dataset, which contains:

1000 audio files

10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock

Each genre has 100 audio files, 30 seconds long, in .wav format

Link: GTZAN Genre Classification - Kaggle

🛠️ Project Pipeline
Audio Preprocessing

Each .wav audio file is converted into a Mel Spectrogram using librosa.

These spectrograms are saved as image files, since CNNs are well-suited to image classification tasks.

Data Preparation

The image dataset (spectrograms) is split into training, validation, and test sets.

Images are resized and normalized for model input (typically to 128x128 or 224x224 pixels).

CNN Model Architecture
A custom Convolutional Neural Network is built using TensorFlow/Keras:

Multiple Conv2D layers with ReLU activation and MaxPooling2D

Dropout layers for regularization

Flattened feature maps connected to a fully connected dense output layer (softmax) for 10-class classification


Training

The model is trained on the spectrogram images using categorical_crossentropy loss and Adam optimizer.

Evaluation & Prediction

Evaluate on the test set to calculate final accuracy.

Given a new song, convert it to a spectrogram → feed into the model → predict the genre.

📊 Results
Training Accuracy: ~98%

Validation Accuracy: ~88.4%

Confusion matrix shows strong classification performance across genres like classical and pop.

Slight confusion between similar genres (e.g., rock vs. metal).

🧠 Why CNNs Work Well Here
CNNs learn spatial patterns in images. Since spectrograms visually represent the frequency content of audio over time, CNNs can pick up on temporal and spectral features—like rhythm, pitch, and texture—making them ideal for genre classification.
