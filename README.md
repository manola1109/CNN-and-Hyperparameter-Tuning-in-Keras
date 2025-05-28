# CNN and Hyperparameter Tuning in Keras

![Last Updated](https://img.shields.io/badge/last%20updated-2025--05--28-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Keras](https://img.shields.io/badge/Keras-latest-red)

This repository contains a comprehensive implementation of Convolutional Neural Networks (CNN) and demonstrates hyperparameter tuning techniques using Keras. The project is designed to provide practical insights into building and optimizing CNN models for image classification tasks.

## Project Overview

This project focuses on:
- Building and training CNNs using Keras
- Implementing various hyperparameter tuning strategies
- Analyzing model performance with different configurations
- Providing practical examples for deep learning practitioners

## Installation

```bash
git clone https://github.com/manola1109/CNN-and-Hyperparameter-Tuning-in-Keras.git
cd CNN-and-Hyperparameter-Tuning-in-Keras
```

### Prerequisites
- Python 3.x
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

## Technical Details

### CNN Architecture

The implemented CNN architecture consists of the following layers:

```python
# Example CNN Architecture
model = Sequential([
    # Input Layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    
    # First Convolutional Block
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Second Convolutional Block
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    # Flatten and Dense Layers
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

### Key Components

1. **Convolutional Layers**
   - Filter sizes: 3x3
   - Number of filters: 32, 64, 128
   - Activation: ReLU
   - Purpose: Feature extraction from input images

2. **Batch Normalization**
   ```python
   # Example of adding batch normalization
   model.add(Conv2D(32, (3, 3), activation='relu'))
   model.add(BatchNormalization())
   ```
   - Stabilizes training
   - Reduces internal covariate shift
   - Improves gradient flow

3. **MaxPooling Layers**
   ```python
   # Example of max pooling implementation
   model.add(MaxPooling2D(pool_size=(2, 2)))
   ```
   - Pool size: 2x2
   - Stride: 2
   - Purpose: Dimension reduction and feature selection

4. **Dropout Layers**
   ```python
   # Example of dropout implementation
   model.add(Dropout(0.25))  # 25% dropout rate
   ```
   - Rates: 0.25 (convolutional layers), 0.5 (dense layers)
   - Purpose: Prevent overfitting

### Hyperparameter Tuning

```python
# Example of hyperparameter tuning setup
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'epochs': [10, 20, 30],
    'optimizer': ['adam', 'rmsprop'],
    'dropout_rate': [0.25, 0.5]
}

# Grid Search Implementation
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn=create_model)
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
```

Key hyperparameters explored:
- Learning rate: 0.001 - 0.1
- Batch size: 32 - 128
- Optimizer choices: Adam, RMSprop
- Dropout rates: 0.25 - 0.5

### Model Training

```python
# Example of model compilation and training
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=20,
    validation_data=(x_val, y_val),
    callbacks=[
        EarlyStopping(patience=5),
        ReduceLROnPlateau(factor=0.1, patience=3),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)
```

### Model Evaluation

```python
# Example of model evaluation
def evaluate_model(model, x_test, y_test):
    # Evaluate the model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {scores[1]*100:.2f}%")
    
    # Generate predictions
    y_pred = model.predict(x_test)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
```

## Project Structure

```
├── CNN and Hyperparameter tuning in keras.ipynb   # Main notebook with implementation
├── README.md                                      # Project documentation
└── LICENSE                                        # MIT License
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook "CNN and Hyperparameter tuning in keras.ipynb"
```

2. Follow the step-by-step implementation in the notebook to:
   - Load and preprocess the dataset
   - Build the CNN model
   - Configure hyperparameters
   - Train and evaluate the model
   - Analyze results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Author

**Deepak Singh Manola** - [@manola1109](https://github.com/manola1109)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow and Keras documentation
- Deep learning community
- Open source contributors

---
Last updated: 2025-05-28 13:33:19 UTC
