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
from jinja2 import ChoiceLoader, FileSystemLoader

# Initialize Flask App
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder="templates")

template_search_paths = [
    os.path.join(BASE_DIR, "templates"),
    BASE_DIR,
]

app.jinja_loader = ChoiceLoader(
    [
        FileSystemLoader(template_search_paths),
        app.jinja_loader,
    ]
)

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

TOOLS = {
    "get_state": {
        "required_args": [],
        "enum_args": {},
    },
    "set_device_state": {
        "required_args": ["device", "state"],
        "enum_args": {
            "device": ["camera", "light", "lock"],
            "state": ["Off", "On", "Locked", "Unlocked"],
        },
    },
}


def invoke_tool(tool_name, args=None):
    args = args or {}
    tool_meta = TOOLS.get(tool_name)

    if tool_meta is None:
        return {
            "ok": False,
            "source": "tool-dispatch",
            "error": "unknown_tool",
            "tool_name": tool_name,
        }

    missing_args = [arg for arg in tool_meta["required_args"] if arg not in args]
    if missing_args:
        return {
            "ok": False,
            "source": "tool-dispatch",
            "error": "missing_required_args",
            "missing": missing_args,
            "tool_name": tool_name,
        }

    for arg_name, allowed_values in tool_meta["enum_args"].items():
        if arg_name in args and args[arg_name] not in allowed_values:
            return {
                "ok": False,
                "source": "tool-dispatch",
                "error": "invalid_enum_value",
                "arg": arg_name,
                "allowed": allowed_values,
                "received": args[arg_name],
                "tool_name": tool_name,
            }

    try:
        if tool_name == "get_state":
            state = esp_network.get_state()
            return {
                "ok": True,
                "source": state.get("source", "unknown"),
                "payload": {
                    "metrics": state.get("metrics", {}),
                    "devices": state.get("devices", {}),
                },
            }

        if tool_name == "set_device_state":
            result = esp_network.set_device_state(args["device"], args["state"])
            return {
                "ok": bool(result.get("ok", True)),
                "source": result.get("source", "unknown"),
                "payload": {
                    "device": result.get("device"),
                    "state": result.get("state"),
                },
            }
    except ValueError as err:
        return {
            "ok": False,
            "source": "tool-dispatch",
            "error": "validation_error",
            "message": str(err),
            "tool_name": tool_name,
        }

    return {
        "ok": False,
        "source": "tool-dispatch",
        "error": "unhandled_tool",
        "tool_name": tool_name,
    }


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
    result = invoke_tool("get_state")
    if not result.get("ok"):
        return jsonify(result), 400
    return jsonify(result)


@app.route("/control/device", methods=["POST"])
def control_device():
    payload = request.get_json(silent=True) or {}
    result = invoke_tool("set_device_state", payload)
    if not result.get("ok"):
        return jsonify(result), 400
    return jsonify(result)


if __name__ == "__main__":
    print("Starting HomeAssistant...")
    app.run(host="0.0.0.0", port=5050, threaded=True)
