## Sentiment Prediction using Neural Networks

### Overview
The main objective of this project was to develop a model that could accurately predict the sentiment behind a text statement. 

There are a total of 5 possible sentimetns that the statement can fall into:
- "sadness"
- "anger"
- "love"
- "surprise"
- "fear"
- "happy"

---

### Model Description

This project includes a neural network model designed for multi-class classification tasks. The model architecture is as follows:

1. **Input Layer**: 
   - 128 neurons
   - ReLU activation function
   - L2 regularization (0.001)

2. **Dropout Layer**: 
   - 50% dropout rate

3. **Hidden Layer**: 
   - 64 neurons
   - ReLU activation function

4. **Dropout Layer**: 
   - 20% dropout rate

5. **Output Layer**: 
   - Number of neurons corresponds to the number of output classes
   - Softmax activation function

The final model is compiled with the RMSprop optimizer and uses categorical crossentropy as the loss function. It also tracks accuracy as a performance metric.

```python
def build_reg_drop_model(output_options_len=10, input_shape=(10000, )):
    # Define the model
    model = models.Sequential()
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.001), activation="relu", input_shape=input_shape))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(output_options_len, activation="softmax"))

    model.compile(optimizer="rmsprop",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model
```

---

### Dataset Used
- **Size** - The dataset consists of a collection of 21 460 entries.
- **Data Types / Structure** - The dataset consist of a singular CSV file with two string columns.
- **Data Source** - The dataset was accessed via Kaggle at [this link]([https://www.kaggle.com/datasets/saurabhshahane/ecommerce-text-classification](https://www.kaggle.com/datasets/ishantjuyal/emotions-in-text?resource=download)).
