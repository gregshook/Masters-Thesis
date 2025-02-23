# Function to get top-k predictions
def get_topk(inputs, tokenizer, model, k=10):
    tokenizer = tokenizer
    model = model 
    def get_layer_hidden_states(input_text):
        # Tokenize the input
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        # Define a forward hook to capture hidden states at each layer
        def hook_fn(module, input, output):
            # Append the output (hidden state) of each layer
            layer_hidden_states.append(output)

        # Attach the hook to each layer
        layer_hidden_states = []
        for layer in model.model.layers:
            layer.register_forward_hook(hook_fn)

        # Forward pass through the model
        with torch.no_grad():
            _ = model(**inputs)

        return layer_hidden_states, inputs

    # Check if input is a single string or a list of strings
    if isinstance(inputs, str):
        inputs = [inputs]  # Convert to list for uniform processing

    results = []  # To store results for all prompts

    for entry in inputs:
        layer_hidden_states, tokenized_inputs = get_layer_hidden_states(entry)

        # Process hidden states for each layer
        for i, hidden_state in enumerate(layer_hidden_states):
            if isinstance(hidden_state, tuple):
                hidden_state = hidden_state[0]

            # Only consider the hidden states of the last token
            hidden_state_last_token = hidden_state[:, -1, :]

            # Project to vocabulary space
            logits = hidden_state_last_token @ model.lm_head.weight.T
            # Perform softmax
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # Get top-k tokens
            top_k_tokens = torch.topk(probs, k, dim=-1).indices[0]
            top_k_tokens_list = top_k_tokens.tolist()

            # Decode the token IDs to words
            top_k_words = [tokenizer.decode([token]) for token in top_k_tokens_list[:k]]

            results.append({
                "Prompt": entry,
                "Layer": i + 1,
                "Top Predictions": top_k_words
            })

    return results

# Function to print top-k results
def print_topk(results):
    for result in results:
        print(f"Prompt: {result['Prompt']}")
        print(f"Layer {result['Layer']} Top Predictions:")
        print("  ".join(result['Top Predictions']))
        print()

# Function to save results to a CSV file
def save_to_file(file_name, results):
    csv_file = str(file_name)
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Prompt", "Layer", "Top Predictions"])
        writer.writeheader()
        for result in results:
            writer.writerow({
                "Prompt": result["Prompt"],
                "Layer": result["Layer"],
                "Top Predictions": ", ".join(result["Top Predictions"])
            })

    print(f"Results saved to {csv_file}")
