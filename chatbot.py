import json
import pickle
import random
import os
import re
import nltk
import numpy as np
import tkinter as tk

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

if os.path.exists('words.pkl'):
    words = pickle.load(open('words.pkl','rb'))
else:
    words = []

if os.path.exists('classes.pkl'):
    classes = pickle.load(open('classes.pkl','rb'))
else:
    classes = []

intents = json.loads(open('intents.json').read())

GREETING_WORDS = {"hi", "hello", "hey", "good morning", "good evening"}
THANK_YOU_WORDS = {"thanks", "thank you", "thx"}
GOODBYE_WORDS = {"bye", "goodbye", "see you", "see ya"}


def contains_any(text, keywords):
    return any(keyword in text for keyword in keywords)


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def normalize_text(sentence):
    return re.findall(r"[a-z0-9']+", sentence.lower())


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)

    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    user_words = set(normalize_text(sentence))
    best_tag = "unknown"
    best_score = 0

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            pattern_words = set(normalize_text(pattern))
            if sentence.lower() in pattern.lower() or pattern.lower() in sentence.lower():
                return intent['tag']

            overlap_score = len(user_words.intersection(pattern_words))
            if overlap_score > best_score:
                best_score = overlap_score
                best_tag = intent['tag']

    if best_score >= 2:
        return best_tag
    
    return "unknown"


def generate_fallback_response(sentence):
    text = sentence.lower().strip()

    if contains_any(text, GREETING_WORDS):
        return random.choice([
            "Hello! How can I help you today?",
            "Hi there. What would you like to know?",
            "Hey. Ask me about products, delivery, refunds, or contact details."
        ])

    if contains_any(text, THANK_YOU_WORDS):
        return random.choice([
            "You're welcome.",
            "Glad I could help.",
            "No problem."
        ])

    if contains_any(text, GOODBYE_WORDS):
        return random.choice([
            "Goodbye! Have a nice day.",
            "See you later.",
            "Take care."
        ])

    if "refund" in text or "return" in text:
        return random.choice([
            "You can return products within 7 days.",
            "Refunds are available for eligible returns within 7 days.",
            "If your item qualifies, you can request a return within 7 days."
        ])

    if "delivery" in text or "shipping" in text or "arrive" in text or "order" in text:
        return random.choice([
            "Delivery usually takes 3-5 business days.",
            "Most orders arrive within 3-5 business days.",
            "Shipping normally takes about 3-5 business days."
        ])

    if "product" in text or "sell" in text or "available" in text:
        return random.choice([
            "We sell electronics, clothing, and accessories.",
            "Our catalog includes electronics, clothing, and accessories.",
            "We currently offer electronics, clothing, and accessories."
        ])

    if "contact" in text or "support" in text or "help" in text or "care" in text:
        return random.choice([
            "You can contact us at support@company.com.",
            "Reach our support team at support@company.com.",
            "For help, email support@company.com."
        ])

    if text.startswith(("who", "what", "when", "where", "why", "how", "which", "can", "could", "do", "does", "is", "are", "will")):
        return random.choice([
            "I can answer product, delivery, refund, and contact questions.",
            "Please give me a little more detail so I can answer properly.",
            "I’m not sure yet, but I can help with support, shipping, refunds, and products."
        ])

    return random.choice([
        "I can help with products, delivery, refunds, and contact details.",
        "Please ask about products, shipping, returns, or support.",
        "I’m not sure about that yet. Try rephrasing your question."
    ])


def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

    return None


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("1.0",tk.END)

    if msg != '':
        ChatLog.config(state=tk.NORMAL)
        ChatLog.insert(tk.END, "You: " + msg + '\n\n')

        tag = predict_class(msg)
        res = get_response(tag)
        if res is None:
            res = generate_fallback_response(msg)

        ChatLog.insert(tk.END, "Bot: " + res + '\n\n')
        ChatLog.config(state=tk.DISABLED)
        ChatLog.yview(tk.END)


base = tk.Tk()
base.title("Customer Service Chatbot")
base.geometry("500x600")

ChatLog = tk.Text(base, bd=0, bg="white", height="8", width="50", font="Arial")

ChatLog.config(state=tk.DISABLED)

scrollbar = tk.Scrollbar(base, command=ChatLog.yview)
ChatLog['yscrollcommand'] = scrollbar.set

SendButton = tk.Button(base, font=("Arial",12,'bold'),
                       text="Send", width="12", height=1,
                       command=send)

EntryBox = tk.Text(base, bd=0, bg="white",width="29", height="5", font="Arial")

scrollbar.place(x=470,y=6, height=500)
ChatLog.place(x=6,y=6, height=500, width=460)
EntryBox.place(x=6, y=520, height=60, width=360)
SendButton.place(x=370, y=520, height=60)

base.mainloop()