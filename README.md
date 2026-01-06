Cassandra — Local Desktop AI Chatbot (Multi-chat GUI) for Windows 10
1. What the application is

Cassandra (multi-chat GUI) is a local desktop AI chatbot for Windows 10, implemented in Python with a Qt GUI (PySide6).
This version is the multi-conversation (multi-chat) GUI variant of Cassandra — it stores and manages multiple conversation files on disk and is not a monolith: the project contains several chatbot variants (terminal, single-chat GUI, multi-chat GUI, context-aware variant for trauma-aware interactions, etc.). This repository/file is the GUI multi-chat version.

Key design decisions (from the code you provided):

Local LLM backend: Ollama (local)

Both text and voice input (record + Whisper or SpeechRecognition fallback)

Both text and voice output (pyttsx3 or Microsoft Edge TTS)

Conversations are saved as JSON files inside a conversations folder next to the script

Conversation files are named like: <safe_title>_<id>.json (title cleaned, plus UUID)

Not monolithic — multiple chatbot variants exist and this GUI focuses on multiple independent conversations and low-resource operation

2. Platform / Testing

Operating System: Windows 10
This application was developed and tested only on Windows 10.
(Do not state or promise support for other OSes unless you test them.)

3. Minimum Python version

Minimum Python: 3.9

4. Features (as implemented in the code)

Multiple conversations (create / rename / delete / save / load)

Conversation history trimmed and saved to disk

Text input and recorded voice input (record → WAV → transcription)

Model conversation context built from recent messages (history limited)

Sends trimmed conversation history to Ollama for assistant replies

Replies appended to conversation and persisted

Voice output option with selectable voice:

edge-tts (Microsoft Edge TTS voices)

pyttsx3 (local/system voices)

Threaded background workers for Ollama calls, recording/transcription, and TTS to keep UI responsive

Conversation and UI limits (to reduce memory / disk use and keep the app low-resource)

Relevant limit constants from the code:

HISTORY_MAX_MESSAGES = 30 (messages sent to the model)

MAX_CHARS_PER_MESSAGE = 3000

SAVE_MAX_MESSAGES = 800 (max messages retained when saving)

UI_MAX_MESSAGES_DISPLAY = 120 (messages shown in the UI)

MAX_TOKENS_DEFAULT = 150 (default max tokens for model calls)

5. Required software & dependencies
Required (core)

Python 3.9+

Ollama (local LLM runner) — installed and running locally

Python packages used in the code:

PySide6

ollama

sounddevice

soundfile

SpeechRecognition

pyttsx3

Optional (recommended for better voice / transcription)

edge-tts (for Microsoft Edge TTS voices)

openai-whisper (for local Whisper transcription, optional — fallback is Google SpeechRecognition)

FFmpeg (may be required for Whisper)

6. Installation — CMD copy/paste

Recommended: create and activate a virtual environment first.

Create + activate virtual environment

python -m venv venv
venv\Scripts\activate


Install required Python packages

pip install PySide6 ollama sounddevice soundfile SpeechRecognition pyttsx3


Optional packages (Edge TTS + Whisper)

pip install edge-tts
pip install openai-whisper


(Optional) If you use Whisper, install ffmpeg and ensure it is on PATH.

7. Ollama setup — CMD copy/paste

Make sure Ollama is installed and running and that at least one model is available locally. Example commands:

# start Ollama server (depends on your Ollama install)
ollama serve

# pull an example model (run if you want a local model)
ollama pull llama3.2


The application uses the local Ollama client and auto-selects a model if present. If no Ollama model/client is available, the GUI will fall back to simple echo behavior (see code).

8. Running the application

Save the script file (the filename is whatever you choose; do not assume main.py). From the folder containing the script:

python your_script_filename.py


Notes:

The GUI will create a conversations folder next to the script (if it does not exist) and save JSON conversation files there.

Conversations are trimmed when very large (see SAVE_MAX_MESSAGES) to avoid huge files.

9. Conversation storage details

Conversations are stored in the conversations directory next to the script.

File naming: <safe_title>_<id>.json — title is sanitized to a safe filename and the UUID is appended.

When saving, the code trims older messages to keep saved files bounded (limit defined by SAVE_MAX_MESSAGES).
