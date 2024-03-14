import random
import json
import torch
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

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

bot_name = "BU-AI"
print("""Hello there, aspiring minds of computer studies! 
      Welcome to BU-AI, your friendly guide to all things related to the world of computer studies. 🌐🤖 
      Whether you're on the hunt for available slots, eager to know about class schedules, keen to explore various subjects, 
      or even curious about the brilliant minds who teach them, BU-AI is your dedicated companion throughout this academic journey. 
      Feel free to ask away – I'm here to help you navigate through your questions! 🎓💻📚 type 'quit' to exit""")
while True:
    sentence = input('You: ')
    if sentence == "quit":
        break
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"""{bot_name}: Oops, it looks like I'm encountering a bit of a brain freeze. 🧠❄️ 
              I'm BU-AI, your trusty computer studies assistant, but it seems I didn't quite catch what you just said. 
              Could you rephrase or try asking your question in a different way? 
              I'm here to help you out with all things related to slots, schedules, subjects, and professors. Don't hesitate to give it another shot! 🤖📚🔍""")


    
    