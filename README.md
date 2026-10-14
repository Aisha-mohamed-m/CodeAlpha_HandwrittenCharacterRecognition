# CodeAlpha_HandwrittenCharacterRecognition

Handwritten digit recognition with a Convolutional Neural Network (CNN) on the MNIST dataset.
Built for **CodeAlpha Machine Learning Internship, Task 3: Handwritten Character Recognition**.

## Objective
Identify handwritten digits (0 to 9) from 28×28 grayscale images using image preprocessing and deep learning.

## Dataset
**MNIST**: 60,000 training and 10,000 test images of handwritten digits. It is loaded automatically with `tf.keras.datasets.mnist`, so no file needs to be downloaded by hand. The classes are balanced (about 6,000 images per digit).

## Project steps
1. Load the data and explore it (sample images, class balance).
2. **Preprocessing:** reshape to `(28, 28, 1)` and normalize pixels to the range 0 to 1.
3. **Model:** CNN with two Conv2D + MaxPooling blocks, a Dense(128) layer and a softmax output (421,642 parameters).
4. **Training:** Adam optimizer, sparse categorical cross-entropy, 10 epochs, batch size 64, 10% validation split.
5. **Evaluation:** test accuracy, confusion matrix, classification report, training curves, error analysis.
6. **Optional improvement:** Dropout and early stopping to reduce overfitting.
7. Save the model as `mnist_cnn.keras`.
8. **Test with your own image:** a preprocessing function converts any photo or drawing of a digit (white, black or coloured background, any ink colour) into a 28×28 MNIST-style input and predicts it.

## CNN architecture

| Layer | Output shape |
|---|---|
| Input | 28 × 28 × 1 |
| Conv2D (32 filters, 3×3, ReLU) | 28 × 28 × 32 |
| MaxPooling2D (2×2) | 14 × 14 × 32 |
| Conv2D (64 filters, 3×3, ReLU) | 14 × 14 × 64 |
| MaxPooling2D (2×2) | 7 × 7 × 64 |
| Flatten | 3136 |
| Dense (128, ReLU) | 128 |
| Dense (10, softmax) | 10 |

## Results

| Metric | Value |
|---|---|
| **Test accuracy** | **99.01%** |
| Test accuracy with Dropout + early stopping | **99.29%** |
| Test loss | 0.043 |
| Final training accuracy | 99.77% |
| Final validation accuracy | 99.10% |
| Precision and recall per digit | 0.98 to 1.00 |
| Misclassified test images | 99 of 10,000 |

- The most frequent mistakes are `5` read as `3` (11 times), `9` as `4` (8 times), `9` as `7` (7 times) and `2` as `7` (6 times).
- There is a small amount of overfitting (training 99.8% vs validation 99.1%). Dropout and early stopping reduce it, and the test accuracy shows that the model generalises well.
- On 5 phone photos of my own handwriting (digits 2, 6, 5, 8 and 3, with shadows and uneven light) the model predicted all 5 correctly.

## Testing your own handwriting
Set `IMAGE_PATH` in the last section of the notebook and run it. No special background or perfect photo is required. The function:
1. finds **thin strokes that are darker (or lighter) than their local surroundings**, so shadows, dark corners, uneven lighting and the colour of the paper or pen do not matter;
2. keeps the pen strokes, ignores pieces that touch the image border or look like long lines, and picks the piece that looks like a digit;
3. thickens thin pen strokes to about the thickness of MNIST strokes, resizes the digit to 20 px and centres it on a 28×28 canvas by its centre of mass, as MNIST does;
4. prints the predicted digit with its confidence and the next two most likely digits, and shows the detection steps.

For best results write **one digit**, keep it away from the image border, and use a pen that is not too faint. The model only knows the digits 0 to 9; letters need EMNIST.

## Files
- `CodeAlpha_MNIST_CNN.ipynb`: full notebook
- `mnist_cnn.keras`: trained model (created when the notebook is run)
- `my_digit.png`: (optional) your own image to test
- `README.md`: this file

## How to run
```bash
pip install tensorflow matplotlib seaborn scikit-learn scipy pillow jupyter
jupyter notebook CodeAlpha_MNIST_CNN.ipynb
```
Training takes about five minutes on a CPU.

## Possible improvements
- Data augmentation (small rotations and shifts).
- Use **EMNIST** to recognise letters as well as digits.
- Use a **CRNN** (CNN + recurrent layers) to read full words or sentences.

## Author
Aisha, SIMAD University
