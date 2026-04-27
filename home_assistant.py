import json
import os
import random
import signal
import string
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
TOOL_CALL_MAX_DEPTH = 2
TOOL_CALL_TIMEOUT_SECONDS = 20
CONFIRMATION_TTL_SECONDS = 45


class SimulatedESPNetwork:
    def __init__(self):
        self.device_states = {
            "camera": "Off",
            "light": "Off",
            "lock": "Locked",
        }
        self.pending_confirmations = {}
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

    def _generate_confirmation_token(self, length=6):
        alphabet = string.ascii_uppercase + string.digits
        return "".join(random.choice(alphabet) for _ in range(length))

    def _cleanup_expired_confirmations(self):
        now = time.monotonic()
        expired_tokens = [
            token
            for token, entry in self.pending_confirmations.items()
            if entry["expires_at_monotonic"] <= now
        ]
        for token in expired_tokens:
            self.pending_confirmations.pop(token, None)

    def _evaluate_transition_policy(self, device, current_state, requested_state):
        if device == "lock" and current_state == "Locked" and requested_state == "Unlocked":
            return {
                "requires_confirmation": True,
                "policy_reason": "Unlocking the door is a high-risk security transition.",
            }

        return {
            "requires_confirmation": False,
            "policy_reason": "No additional confirmation required by policy.",
        }

    def request_device_state_change(self, device, state):
        if device not in self.device_states:
            raise ValueError(f"Unknown device '{device}'.")

        self._cleanup_expired_confirmations()
        current_state = self.device_states[device]
        policy = self._evaluate_transition_policy(device, current_state, state)

        if policy["requires_confirmation"]:
            token = self._generate_confirmation_token()
            issued_monotonic = time.monotonic()
            expires_monotonic = issued_monotonic + CONFIRMATION_TTL_SECONDS
            issued_unix = int(time.time())
            expires_unix = issued_unix + CONFIRMATION_TTL_SECONDS

            self.pending_confirmations[token] = {
                "device": device,
                "state": state,
                "issued_at_monotonic": issued_monotonic,
                "expires_at_monotonic": expires_monotonic,
                "issued_at_unix": issued_unix,
                "expires_at_unix": expires_unix,
                "policy_reason": policy["policy_reason"],
            }

            return {
                "ok": True,
                "source": "simulated-esp-over-wifi",
                "status": "pending_confirmation",
                "requires_confirmation": True,
                "confirmation_token": token,
                "confirmation_ttl_seconds": CONFIRMATION_TTL_SECONDS,
                "expires_at_unix": expires_unix,
                "device": device,
                "state": state,
                "policy_reason": policy["policy_reason"],
            }

        result = self.set_device_state(device, state)
        result["status"] = "applied"
        result["requires_confirmation"] = False
        result["policy_reason"] = policy["policy_reason"]
        return result

    def confirm_device_state_change(self, token):
        if not token:
            raise ValueError("Confirmation token is required.")

        self._cleanup_expired_confirmations()
        pending = self.pending_confirmations.get(token)
        if not pending:
            return {
                "ok": False,
                "source": "simulated-esp-over-wifi",
                "error": "invalid_or_expired_confirmation_token",
                "status": "confirmation_rejected",
            }

        result = self.set_device_state(pending["device"], pending["state"])
        self.pending_confirmations.pop(token, None)
        result["status"] = "applied"
        result["requires_confirmation"] = False
        result["policy_reason"] = pending["policy_reason"]
        return result


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
    "confirm_device_state": {
        "required_args": ["confirmation_token"],
        "enum_args": {},
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
            result = esp_network.request_device_state_change(args["device"], args["state"])
            payload = {
                "device": result.get("device"),
                "state": result.get("state"),
                "status": result.get("status", "applied"),
                "requires_confirmation": bool(result.get("requires_confirmation", False)),
                "policy_reason": result.get("policy_reason"),
            }
            if result.get("status") == "pending_confirmation":
                payload["confirmation_token"] = result.get("confirmation_token")
                payload["confirmation_ttl_seconds"] = result.get("confirmation_ttl_seconds")
                payload["expires_at_unix"] = result.get("expires_at_unix")
            return {
                "ok": bool(result.get("ok", True)),
                "source": result.get("source", "unknown"),
                "payload": payload,
            }

        if tool_name == "confirm_device_state":
            result = esp_network.confirm_device_state_change(args["confirmation_token"])
            response = {
                "ok": bool(result.get("ok", False)),
                "source": result.get("source", "unknown"),
                "payload": {
                    "status": result.get("status"),
                    "device": result.get("device"),
                    "state": result.get("state"),
                    "requires_confirmation": bool(result.get("requires_confirmation", False)),
                    "policy_reason": result.get("policy_reason"),
                },
            }
            if not response["ok"]:
                response["error"] = result.get("error")
            return response
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


def build_tool_or_answer_prompt(user_input, tool_history):
    tool_history_json = json.dumps(tool_history, indent=2)
    return f"""You are Daniel's personal HomeAssistant running locally on a Raspberry Pi.
You can either answer directly or ask for one tool call.
Available tools:
- get_state: takes no arguments.
- set_device_state: args {{"device": "camera|light|lock", "state": "Off|On|Locked|Unlocked"}}
- confirm_device_state: args {{"confirmation_token": "echo token from pending confirmation"}}

Conversation user request:
{user_input}

Tool history (JSON):
{tool_history_json}

Return ONLY valid JSON with one of these shapes:
1) {{"type":"final_answer","answer":"concise plain text"}}
2) {{"type":"tool_call","tool_name":"get_state|set_device_state|confirm_device_state","args":{{...}}}}

Do not include markdown or extra keys.
"""


def build_final_answer_prompt(user_input, tool_history):
    tool_history_json = json.dumps(tool_history, indent=2)
    return f"""You are Daniel's personal HomeAssistant running locally on a Raspberry Pi.
Provide a concise final answer for the user.

User request:
{user_input}

Tool results (JSON):
{tool_history_json}

Return plain text only.
"""


def get_response(prompt, model=OLLAMA_MODEL, temperature=0.7, max_tokens=500):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
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


def run_agent_flow(user_input):
    tool_history = []
    start_time = time.monotonic()

    for _ in range(TOOL_CALL_MAX_DEPTH + 1):
        if time.monotonic() - start_time > TOOL_CALL_TIMEOUT_SECONDS:
            return "I couldn't finish safely in time. Please try again."

        first_pass_raw = get_response(
            build_tool_or_answer_prompt(user_input, tool_history),
            temperature=0.1,
            max_tokens=280,
        )

        try:
            first_pass = json.loads(first_pass_raw)
        except json.JSONDecodeError:
            return first_pass_raw

        if first_pass.get("type") == "final_answer":
            return str(first_pass.get("answer", "")).strip() or "Okay."

        if first_pass.get("type") != "tool_call":
            return "I couldn't determine the next step."

        if len(tool_history) >= TOOL_CALL_MAX_DEPTH:
            return "I hit the maximum tool-call depth. Please refine your request."

        tool_name = first_pass.get("tool_name")
        args = first_pass.get("args") if isinstance(first_pass.get("args"), dict) else {}
        tool_result = invoke_tool(tool_name, args)
        tool_history.append(
            {
                "tool_name": tool_name,
                "args": args,
                "result": tool_result,
            }
        )

        second_pass = get_response(
            build_final_answer_prompt(user_input, tool_history),
            temperature=0.2,
            max_tokens=220,
        )

        if second_pass and "Ollama" not in second_pass:
            return second_pass.strip()

    return "I couldn't complete that request."


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

    response = run_agent_flow(user_input)
    speak(response)
    return jsonify({"response": response})


@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    user_input = request.json.get("input")

    if not user_input:
        return jsonify({"response": "No input received."}), 400

    def generate():
        try:
            full_response = run_agent_flow(user_input)

            for token in full_response.split(" "):
                chunk = f"{token} "
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"

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
    if result.get("payload", {}).get("status") == "pending_confirmation":
        return jsonify(result), 202
    return jsonify(result)


@app.route("/control/confirm", methods=["POST"])
def control_confirm():
    payload = request.get_json(silent=True) or {}
    result = invoke_tool("confirm_device_state", payload)
    if not result.get("ok"):
        return jsonify(result), 400
    return jsonify(result)


if __name__ == "__main__":
    print("Starting HomeAssistant...")
    app.run(host="0.0.0.0", port=5050, threaded=True)
