
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
from transformers import MarianMTModel, MarianTokenizer
from googletrans import Translator
from difflib import SequenceMatcher

app = Flask(__name__)
run_with_ngrok(app)

model_name = "Helsinki-NLP/opus-mt-en-de"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

translator = Translator()

def hf_translate(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def google_translate(text):
    return translator.translate(text, dest="de").text

def similarity(a,b):
    return SequenceMatcher(None,a,b).ratio()

@app.route("/", methods=["GET","POST"])

def home():

    hf=""
    google=""
    score=0
    flag=False

    if request.method=="POST":

        text=request.form["text"]

        hf=hf_translate(text)
        google=google_translate(text)

        score=similarity(hf,google)

        if score<0.75:
            flag=True

    return render_template("index.html",
                           hf=hf,
                           google=google,
                           score=round(score*100,2),
                           flag=flag)

if __name__=="__main__":
    app.run()
