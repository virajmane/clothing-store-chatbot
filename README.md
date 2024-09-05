# Clothing Store Chatbot

## Overview

This project involves building a chatbot for a clothing store using a fine-tuned GPT-2 model. The chatbot is designed to answer customer queries related to products, services, and store policies. It also handles out-of-scope questions by suggesting related queries.

## Features

- **Fine-tuned GPT-2 Model**: Custom-trained to respond to store-related queries.
- **Scoped Responses**: The bot is instructed to only answer questions about the store and its offerings.
- **Graceful Handling of Out-of-Scope Questions**: Provides a default response when a question is outside its scope.

## Prerequisites

- Python 3.7+
- `transformers` library
- `torch` library

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/clothing-store-chatbot.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd clothing-store-chatbot
   ```

3. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Files

- `finetune.py`: Script to fine-tune the GPT-2 model on the custom training data.
- `evaluate.py`: Main script to load the model, generate responses, and handle user queries.
- `requirements.txt`: Lists the dependencies required for the project.
- `data.txt`: Contains the training data used to fine-tune the model.

## Usage

### Fine-tuning the Model

1. **Prepare the data**:
   Ensure that `data.txt` is properly formatted with your training data.

2. **Run the fine-tuning script**:
   ```bash
   python finetune.py
   ```
   This will fine-tune the GPT-2 model on your custom data and save the fine-tuned model to the `./fine_tuned_model` directory.

### Running the Chatbot

1. **Prepare the model**:
   Ensure that your fine-tuned model and tokenizer are saved in the `./fine_tuned_model` directory.

2. **Run the chatbot script**:
   ```bash
   python evaluate.py
   ```

   You will be prompted to ask questions. The bot will respond based on the fine-tuned model and instructions provided.

### Example Interaction

When you run `evaluate.py`, you can interact with the bot as follows:

```
Ask me: Can I track my order?
Bot: Yes, you can track your order using the tracking number provided by your payment processor.

Ask me: What is HTML?
Bot: This question is not within my scope. Please ask about products, orders, or store-related queries.
```

## Code Explanation

- **`finetune.py`**: This script fine-tunes the GPT-2 model using the data in `data.txt`. It prepares the data, sets up the training loop, and saves the fine-tuned model.

   ```python
   from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
   import pandas as pd

   # Load the pre-trained model and tokenizer
   model_name = "gpt2"
   model = GPT2LMHeadModel.from_pretrained(model_name)
   tokenizer = GPT2Tokenizer.from_pretrained(model_name)

   # Load the dataset
   dataset = pd.read_csv('data.txt', delimiter='\t', header=None, names=['input', 'output'])

   # Tokenize the data
   def tokenize_function(examples):
       return tokenizer(examples['input'], truncation=True, padding='max_length')

   tokenized_datasets = dataset.apply(tokenize_function, axis=1)

   # Define training arguments
   training_args = TrainingArguments(
       per_device_train_batch_size=4,
       num_train_epochs=3,
       logging_dir='./logs',
   )

   trainer = Trainer(
       model=model,
       args=training_args,
       train_dataset=tokenized_datasets,
   )

   # Train the model
   trainer.train()

   # Save the model
   model.save_pretrained('./fine_tuned_model')
   tokenizer.save_pretrained('./fine_tuned_model')
   ```

- **`evaluate.py`**: This script loads the fine-tuned GPT-2 model and tokenizer, generates responses based on user input, and handles out-of-scope questions by suggesting related queries.

   ```python
   from transformers import GPT2LMHeadModel, GPT2Tokenizer

   # Load fine-tuned model and tokenizer
   model_path = './fine_tuned_model'
   model = GPT2LMHeadModel.from_pretrained(model_path)
   tokenizer = GPT2Tokenizer.from_pretrained(model_path)

   # Set pad_token_id explicitly
   tokenizer.pad_token = tokenizer.eos_token
   model.config.pad_token_id = tokenizer.pad_token_id

   # Define a function for generating responses
   def generate_response(prompt):
       # Encode the input prompt
       inputs = tokenizer.encode(prompt, return_tensors='pt')
       
       # Create attention mask
       attention_mask = inputs.ne(tokenizer.pad_token_id)
       
       # Generate a response
       outputs = model.generate(
           inputs,
           attention_mask=attention_mask,
           max_length=200,
           num_return_sequences=1,
           pad_token_id=tokenizer.pad_token_id  # Set pad token ID
       )
       
       # Decode and return the response
       response = tokenizer.decode(outputs[0], skip_special_tokens=True)
       return response

   # Example usage
   prompt = ""
   while prompt != "q":
       instruction = "You are a chatbot for a clothing store. Only answer questions about our products, services, or store policies. If the question is outside your scope, respond with 'This question is not within my scope. Please ask about products, orders, or store-related queries.'"
       prompt = input("Ask me: ")
       response = generate_response(f"{instruction}\nUser: {prompt}\nBot:")
       print(response)
   ```

## Training Data

The `data.txt` file contains the training data for the fine-tuned model. It includes customer queries and the corresponding responses related to the clothing store.

### Sample Data (`data.txt`)

```
User: What is your return policy?
Bot: We offer a 30-day return policy for all items purchased. Please ensure that items are in their original condition and packaging.

User: Do you offer gift cards?
Bot: Yes, we offer gift cards in various denominations. You can purchase them online or in-store.

User: How do I track my order?
Bot: You can track your order using the tracking number provided by your payment processor.

User: Can you tell me about HTML?
Bot: This question is not within my scope. Please ask about products, orders, or store-related queries.
```

## Notes

- Ensure that the model is properly fine-tuned on relevant data to achieve accurate and relevant responses.
- Customize the instructions and data according to the specific needs of your store.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or contributions, please contact.
```

