from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize Flask app
app = Flask(__name__)

# Load fine-tuned model and tokenizer
model_path = './fine_tuned_model'
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Set pad_token explicitly
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Function to generate a chatbot response
def generate_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs,
        max_length=200,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        no_repeat_ngram_size=2,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to check if the query is within scope
def is_within_scope(prompt):
    allowed_topics = ["products", "orders", "shipping", "returns", "store policies", "promotions"]
    return any(topic in prompt.lower() for topic in allowed_topics)

# Flask route for handling chatbot queries
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data.get('prompt', '')

    # if not is_within_scope(prompt):
    #     return jsonify({"response": "This question is not within my scope. Please ask about products, orders, or store-related queries."})
    
    # instruction = (
    #     "You are a chatbot for a clothing store. "
    #     "Only answer questions about our products, services, or store policies. "
    #     "If the question is outside your scope, respond with 'This question is not within my scope. "
    #     "Please ask about products, orders, or store-related queries.'"
    # )
    
    # full_prompt = f"{instruction}\nUser: {prompt}\nBot:"
    response = generate_response(prompt)
    
    return jsonify({"response": response})

# Route for serving the chat page
@app.route('/')
def index():
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug=True)
