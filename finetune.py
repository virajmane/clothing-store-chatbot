import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# Load dataset from the IPC data provided
def load_ipc_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().split('\n')  # Split into sentences or lines
    return Dataset.from_dict({'text': text})

# Define paths
model_path = 'gpt2'  # Base model, can be replaced with the path to your model
data_path = 'data.txt'  # Your IPC data file

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Set pad_token to eos_token
tokenizer.pad_token = tokenizer.eos_token

# Load and preprocess data
dataset = load_ipc_dataset(data_path)

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, return_tensors='pt')  # Use dynamic padding

# Preprocess the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We are fine-tuning GPT-2, which is a language model (not masked LM)
)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',  # Output directory for model checkpoints
    evaluation_strategy="steps",  # Evaluate every few steps
    eval_steps=500,  # Evaluate every 500 steps
    save_steps=10_000,  # Save checkpoint every 10,000 steps
    per_device_train_batch_size=1,  # Adjust based on your GPU memory
    gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
    per_device_eval_batch_size=1,  # Adjust based on your GPU memory
    num_train_epochs=10,  # Increased number of training epochs
    learning_rate=5e-5,  # Lower learning rate for better convergence
    weight_decay=0.01,  # Weight decay for regularization
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=500,  # Log every 500 steps
    report_to="none",  # Disable reporting to avoid issues
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,  # Train on the tokenized dataset
)

# Start training
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
