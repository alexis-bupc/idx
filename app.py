from flask import Flask, render_template, request, jsonify
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from spellchecker import SpellChecker

app = Flask(__name__)

spell_checker = SpellChecker()

# Load model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
file_path = r'C:\Users\alexi\OneDrive\Desktop\Documents\BU_AI-with-Flask\intents.json' #replace with your own path, yung path ng intents.json sa files mo, right click, copy as path
with open(file_path, 'r', encoding='utf-8') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    if user_input == "quit":
        return jsonify({'response': "Goodbye!"})
    
    # Spell checking
    corrected_words = [spell_checker.correction(word) for word in user_input.split() if spell_checker.correction(word) is not None]
    corrected_input = " ".join(corrected_words)
    
    sentence = tokenize(corrected_input)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                bot_response = random.choice(intent['responses'])
                return jsonify({'response': bot_response})
    else:
        bot_response = [ "I'm sorry, I'm having trouble understanding that.",
                            "My apologies, I didn't catch that. Could you try rephrasing?",
                            "I'm here to assist, but I'm not quite following you. Can you provide more context?",
                            "It seems I'm a bit lost. Could you provide more information?",
                            "I'm here to help, but I'm not sure I understood your question correctly. Could you elaborate?"]
        return jsonify({'response': random.choice(bot_response)})

if __name__ == '__main__':
    app.run(debug=True)
