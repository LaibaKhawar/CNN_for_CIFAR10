# CNN for CIFAR-10 Image Classification

A three-block convolutional network in TensorFlow/Keras trained on CIFAR-10, with a data-augmentation comparison and a learning-rate ablation.

**Case study:** [Read it](https://laiba-khawar-portfolio.vercel.app/work/cifar10-cnn)

## How it works

All code is in `CNN_for_CIFAR10.ipynb`.

**Baseline (cell 4).**

- Loads CIFAR-10 from the Hugging Face Hub (`datasets.load_dataset("cifar10")`), scales pixels to [0, 1] and one-hot encodes the labels.
- Splits the 50,000 training images 80/20 (`random_state=42`) into 40,000 for training and 10,000 for validation.
- Model: Conv2D 32, MaxPool, Conv2D 64, MaxPool, Conv2D 128, MaxPool (all 3x3, ReLU, `same` padding), Flatten, Dense 128 (ReLU), Dropout 0.5, Dense 10 (softmax).
- Adam (default learning rate 0.001), categorical cross-entropy, 10 epochs, batch size 64.
- Evaluated on the official 10,000-image test set, with a classification report, confusion matrix and examples of misclassified images.

**Data augmentation (cells 7 to 10).** Reloads CIFAR-10 through `keras.datasets`, builds an `ImageDataGenerator` (rotation 15°, width/height shift 0.1, horizontal flip, zoom 0.1) and trains for 10 more epochs on augmented batches, then evaluates on the test set.

**Learning-rate ablation (cells 14 and 15).** Builds a fresh copy of the baseline model for each learning rate in {0.001, 0.01, 0.1}, trains for 10 epochs on the 40,000-image split and evaluates on the 10,000-image validation split. Validation accuracy and loss curves are plotted.

**Further ablations (cells 17 to 27).** Batch size (16, 32, 64), number of filters (16, 32, 64) and number of convolutional layers (3, 5), each with training times.

## Results

| Experiment | Test accuracy | Source |
| --- | --- | --- |
| Baseline CNN | 0.7410 | cell 4 (also printed in cell 10) |
| After augmented training | 0.7424 | cell 9 (also printed in cell 10) |

Learning-rate ablation, accuracy on the held-out 10,000-image split (cell 15):

| Learning rate | Accuracy |
| --- | --- |
| 0.001 | 0.7432 |
| 0.01 | 0.4249 |
| 0.1 | 0.1014 (diverged, chance level for 10 classes) |

## Repository contents

- `CNN_for_CIFAR10.ipynb`: all experiments.
- `requirements.txt`: datasets, jupyter, matplotlib, numpy, scikit-learn, seaborn, tensorflow.

## Running it

The notebook was last run locally with Python 3.9 and TensorFlow 2.18 on Windows.

```bash
git clone https://github.com/LaibaKhawar/CNN_for_CIFAR10.git
cd CNN_for_CIFAR10
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook CNN_for_CIFAR10.ipynb
```

CIFAR-10 is downloaded automatically (once from the Hugging Face Hub and once through Keras). Cells 0 to 3 are `pip install` commands from the original environment, including a forced `numpy==1.23.5` downgrade that conflicts with TensorFlow 2.18; skip them and use `requirements.txt` instead.

## Known limitations

- **The augmentation comparison is not independent.** Cell 8 continues training the already-trained baseline `model` for 10 more epochs with augmentation rather than training a new model, so the 0.7424 reflects 20 epochs in total and the gain over 0.7410 cannot be attributed to augmentation alone.
- **The classification report in cell 11 is invalid.** It predicts on `X_test` (the validation split from the Hugging Face data) but compares with `y_test` from the Keras test set, so the labels do not match the images and the report shows chance-level numbers.
- **Later ablation cells reused a diverged model.** The batch-size cell (17) and the filter-count cell (20) call `fit` on the `model` variable left over from the learning-rate loop, which is the diverged LR=0.1 model. Their accuracies are not meaningful. The filter-count loop also never changes the number of filters. The layer-count cell (25) does build new models, but its `create_model` adds a max-pool after each extra conv layer and has no dropout, so it differs from the baseline architecture.
- The learning-rate results are measured on the validation split, not the official test set.
- Single runs, no fixed TensorFlow seed.

## Author

[Laiba Khawar](https://github.com/LaibaKhawar) · [LinkedIn](https://www.linkedin.com/in/laiba-k-00b2b1249/)
