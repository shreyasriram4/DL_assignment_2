# LSTM and GRU Language Modeling
This project implements LSTM and GRU-based neural networks for language modeling using the Penn Tree Bank dataset. The models were trained with and without dropout to explore the effects of regularization. Saved models can be loaded to evaluate their performance.

## Requirements
- Python 3.x
- PyTorch
- NumPy
- pandas
- tqdm

Install the dependencies:
```
pip install torch numpy pandas tqdm
```
## Data Preparation
The Penn Tree Bank (PTB) dataset is required to train/test the models. Make sure the data is stored in the data/ptb_data/ directory. The dataset should include the following files:

- train.txt
- valid.txt
- test.txt

## Training the Models
The models can be trained using the code provided in the main.ipynb file. Depending on the desired configuration (LSTM/GRU with or without dropout), the appropriate sections in the notebook should be executed. Below are steps to train each configuration:

- LSTM Without Dropout:
To train the LSTM without dropout, run the cell in the notebook corresponding to the LSTM model section without adding the dropout layer. The optimizer used is SGD.

- LSTM With Dropout:
To train the LSTM with dropout, execute the section where dropout layers are applied after the embedding and LSTM layers. Use Adam as the optimizer for faster convergence.

- GRU Without Dropout:
For the GRU model without dropout, run the GRU section without applying any dropout layers.

- GRU With Dropout:
To train the GRU with dropout, use the section that includes dropout after the embedding and GRU layers.

Training logs will display the train, validation, and test perplexity over time.

## Testing the Models (Loading Saved Weights)
Each model has been trained and its weights have been saved in the models/ directory. You can load the pre-trained models and test them on the validation or test dataset using the following steps in main.ipynb:

### Load the Pre-Trained Models: Modify the cell to load the desired pre-trained model:

```
import torch
from model import LSTMModel, GRUModel  # assuming models are defined in model.py

# Example: Loading the LSTM model without dropout
model = LSTMModel(vocab_size, embed_size, hidden_size, num_layers)
model.load_state_dict(torch.load('models/lstm_no_dropout.pth'))

# Example: Loading the GRU model with dropout
model = GRUModel(vocab_size, embed_size, hidden_size, num_layers, dropout=0.25)
model.load_state_dict(torch.load('models/gru_with_dropout.pth'))

model.eval()  # Set the model to evaluation mode

```

### Evaluate the Model
Once the model is loaded, you can evaluate its performance on the validation or test dataset:

```
# Evaluate the model on the validation set
validation_perplexity = evaluate(model, validation_data)
print(f'Validation Perplexity: {validation_perplexity}')

# Evaluate the model on the test set
test_perplexity = evaluate(model, test_data)
print(f'Test Perplexity: {test_perplexity}')
```

### Test with Saved Models: 
To test the models with the pre-saved weights, simply change the path to the corresponding .pth file in the load_state_dict() function and run the evaluation script in the notebook.

## Conclusion
This project demonstrates the importance of regularization techniques such as dropout in reducing overfitting and improving generalization in recurrent neural networks. By comparing LSTM and GRU models with and without dropout, it is evident that careful hyperparameter tuning and regularization can lead to significant improvements in language modeling tasks.
