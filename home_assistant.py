import json
import os
import random
import signal
import sys
import time

import pyttsx3
import requests
import speech_recognition as sr
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from flask_cors import cross_origin

# Initialize Flask App
app = Flask(__name__, template_folder=".")

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 0.9)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"


class SimulatedESPNetwork:
    def __init__(self):
        self.device_states = {
            "camera": "Off",
            "light": "Off",
            "lock": "Locked",
        }
        self.metrics = {
            "water_usage_litres": 1560,
            "electricity_usage_kwh": 31.7,
            "weather": "22°C · Clear",
        }

    def _wifi_ping_delay(self):
        time.sleep(0.12)

    def get_state(self):
        self._wifi_ping_delay()
        self.metrics["water_usage_litres"] += random.choice([1, 2, 3, 5])
        self.metrics["electricity_usage_kwh"] = round(
            self.metrics["electricity_usage_kwh"] + random.choice([0.0, 0.1, 0.2]),
            1,
        )
        return {
            "source": "simulated-esp-over-wifi",
            "metrics": self.metrics,
            "devices": self.device_states,
        }

    def set_device_state(self, device, state):
        if device not in self.device_states:
            raise ValueError(f"Unknown device '{device}'.")

        self._wifi_ping_delay()
        self.device_states[device] = state
        return {
            "ok": True,
            "source": "simulated-esp-over-wifi",
            "device": device,
            "state": state,
        }


esp_network = SimulatedESPNetwork()


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


@app.route("/control/state", methods=["GET"])
def control_state():
    return jsonify(esp_network.get_state())


@app.route("/control/device", methods=["POST"])
def control_device():
    payload = request.get_json(silent=True) or {}
    device = payload.get("device")
    state = payload.get("state")

    if not device or not state:
        return jsonify({"error": "Both 'device' and 'state' are required."}), 400

    try:
        result = esp_network.set_device_state(device, state)
        return jsonify(result)
    except ValueError as err:
        return jsonify({"error": str(err)}), 400


if __name__ == "__main__":
    print("Starting HomeAssistant...")
    app.run(host="0.0.0.0", port=5050, threaded=True)
