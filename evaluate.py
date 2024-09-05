from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load fine-tuned model and tokenizer
model_path = './fine_tuned_model'
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Set pad_token explicitly
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Define a function for generating responses
def generate_response(prompt):
    # Encode the input prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate a response with sampling enabled
    outputs = model.generate(
        inputs,
        max_length=200,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,  # Set pad token ID
        do_sample=True,  # Enable sampling
        temperature=0.7,  # Adjust temperature for more focused answers
        top_p=0.9,  # Use nucleus sampling
        top_k=50,  # Restrict number of word options per token
        no_repeat_ngram_size=2,  # Prevent repetitive phrases
    )
    
    # Decode and return the response
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to check if the query is within scope
def is_within_scope(prompt):
    allowed_topics = ["products", "orders", "shipping", "returns", "store policies", "promotions"]
    # Check if any of the allowed topics is in the prompt
    return any(topic in prompt.lower() for topic in allowed_topics)

# Function to handle prompting and limit scope to store-related queries
def handle_user_query():
    prompt = ""
    while prompt != "q":
        instruction = (
            "You are a chatbot for a clothing store. "
            "Only answer questions about our products, services, or store policies. "
            "If the question is outside your scope, respond with 'This question is not within my scope. "
            "Please ask about products, orders, or store-related queries.'"
        )
        prompt = input("Ask me: ")
        
        # Exit command
        if prompt.strip().lower() == "q":
            print("Goodbye!")
            break
        
        # Check if the prompt is within scope
        if not is_within_scope(prompt):
            print("This question is not within my scope. Please ask about products, orders, or store-related queries.")
            continue
        
        # Generate response
        response = generate_response(f"{instruction}\nUser: {prompt}\nBot:")
        print(response)

# Run the chatbot interaction
handle_user_query()
