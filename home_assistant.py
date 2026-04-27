import json
import requests
import pyttsx3
import speech_recognition as sr
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import cross_origin
import os
import sys
import time
import signal

# Initialize Flask App
app = Flask(__name__, template_folder="templates", static_folder="static")

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 0.9)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"


def build_prompt(user_input):
    return f"""You are Daniel's personal HomeAssistant running locally on a Raspberry Pi.
You are direct, practical, and useful.
Do not give generic AI disclaimers.
Keep answers concise unless the user asks for detail.

User: {user_input}
Assistant:"""


def get_response(prompt, model=OLLAMA_MODEL, temperature=0.7, max_tokens=500):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": build_prompt(prompt),
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            },
            timeout=180
        )

        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    except requests.exceptions.ConnectionError:
        return "Ollama is not running."
    except requests.exceptions.Timeout:
        return "Ollama timed out."
    except Exception as e:
        return f"Ollama error: {e}"


def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS error: {e}")


def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        try:
            audio = recognizer.listen(source, timeout=5)
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, I didn't catch that."
        except sr.WaitTimeoutError:
            return "Listening timed out."


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("input")
    if not user_input:
        return jsonify({"response": "No input received."}), 400

    response = get_response(user_input)
    speak(response)
    return jsonify({"response": response})


@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    user_input = request.json.get("input")

    if not user_input:
        return jsonify({"response": "No input received."}), 400

    def generate():
        full_response = ""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": build_prompt(user_input),
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 500
                    }
                },
                stream=True,
                timeout=180
            )

            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                data = json.loads(line.decode("utf-8"))
                chunk = data.get("response", "")

                if chunk:
                    full_response += chunk
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"

                if data.get("done", False):
                    break

            yield f"data: {json.dumps({'done': True})}\n\n"

            if full_response.strip():
                speak(full_response.strip())

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/restart", methods=["POST"])
@cross_origin()
def restart():
    try:
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(1)
        os.execl(sys.executable, sys.executable, *sys.argv)
    except Exception as e:
        return jsonify({"message": f"Failed to restart. Error: {str(e)}"}), 500

    return jsonify({"message": "Service restarted successfully!"})


if __name__ == "__main__":
    print("Starting HomeAssistant...")
    app.run(host="0.0.0.0", port=5050, threaded=True)
