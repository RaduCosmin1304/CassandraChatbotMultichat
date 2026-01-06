#!/usr/bin/env python3
from __future__ import annotations
import sys
import os
import platform
import subprocess
import time
import json
import re
import uuid
import traceback as tbmod
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

from PySide6 import QtCore, QtWidgets, QtGui

HISTORY_MAX_MESSAGES = 30
MAX_CHARS_PER_MESSAGE = 3000
SAVE_MAX_MESSAGES = 800
UI_MAX_MESSAGES_DISPLAY = 120
MAX_TOKENS_DEFAULT = 150
try:
    import ollama
    from ollama import Client, ResponseError  
except Exception:
    ollama = None
    Client = None
    ResponseError = Exception  

try:
    import edge_tts
except Exception:
    edge_tts = None

try:
    import whisper
except Exception:
    whisper = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    import sounddevice as sd
    import soundfile as sf
except Exception:
    sd = None
    sf = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None
_model_re = re.compile(r"model\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
_simple_id_re = re.compile(r"([a-zA-Z0-9_\-]+:[a-zA-Z0-9_\-\.]+)")

def parse_model_from_repr(obj_repr: str) -> Optional[str]:
    m = _model_re.search(obj_repr)
    if m:
        return m.group(1)
    m2 = _simple_id_re.search(obj_repr)
    if m2:
        return m2.group(1)
    return None


def to_model_string(item: Any) -> Optional[str]:
    if isinstance(item, dict):
        for key in ("model", "name", "id"):
            val = item.get(key)
            if isinstance(val, str) and val:
                return val
        return None
    if isinstance(item, str) and item:
        if ":" in item:
            return item
        return parse_model_from_repr(item)
    for attr in ("model", "name", "id"):
        if hasattr(item, attr):
            val = getattr(item, attr)
            if isinstance(val, str) and val:
                return val
    try:
        return parse_model_from_repr(repr(item))
    except Exception:
        return None


def get_installed_models(client: Any) -> List[str]:
    models: List[str] = []
    try:
        if client is not None and hasattr(client, "list"):
            res = client.list()
            if isinstance(res, (list, tuple)):
                for it in res:
                    s = to_model_string(it)
                    if s:
                        models.append(s)
                if models:
                    return models
            if isinstance(res, dict) and "models" in res and isinstance(res["models"], list):
                for it in res["models"]:
                    s = to_model_string(it)
                    if s:
                        models.append(s)
                if models:
                    return models
            s = to_model_string(res)
            if s:
                return [s]
    except Exception:
        pass
    try:
        if ollama is not None and hasattr(ollama, "list"):
            res = ollama.list()
            if isinstance(res, (list, tuple)):
                for it in res:
                    s = to_model_string(it)
                    if s:
                        models.append(s)
                if models:
                    return models
            s = to_model_string(res)
            if s:
                return [s]
    except Exception:
        pass
    return models


def make_ollama_client(host: str = "localhost", port: int = 11434) -> Optional[Any]:
    if Client is None:
        return None
    base = f"http://{host}:{port}"
    try:
        return Client()
    except Exception:
        try:
            return Client(base_url=base)
        except Exception:
            try:
                return Client(host=base)
            except Exception:
                return None

def choose_model_for_client(client: Any, requested: Optional[str] = None) -> Optional[str]:
    installed = get_installed_models(client)
    if not installed:
        return None
    if requested:
        if requested in installed:
            return requested
        for m in installed:
            if m.startswith(requested):
                return m
        return None
    if "llama3.2:latest" in installed:
        return "llama3.2:latest"
    for m in installed:
        if m.startswith("llama3.2"):
            return m
    return installed[0]

def call_ollama_chat(client: Any, model: str, messages: List[Dict[str, str]], max_tokens=200, temperature=0.2):
    opts = {"max_tokens": max_tokens, "temperature": temperature}
    try:
        return client.chat(model=model, messages=messages, options=opts)
    except TypeError:
        try:
            return client.chat(model=model, messages=messages)
        except TypeError:
            return client.chat(model, messages)
    except Exception:
        raise

def extract_text_from_response(resp: Any) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    try:
        if isinstance(resp, dict):
            m = resp.get("message")
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                return m["content"]
            if isinstance(resp.get("response"), str):
                return resp["response"]
            choices = resp.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    if isinstance(first.get("content"), str):
                        return first["content"]
                    mm = first.get("message")
                    if isinstance(mm, dict) and isinstance(mm.get("content"), str):
                        return mm["content"]
        if hasattr(resp, "message"):
            msg = getattr(resp, "message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if hasattr(msg, "content"):
                return getattr(msg, "content")
        for attr in ("response", "content", "text", "output", "result"):
            if hasattr(resp, attr):
                v = getattr(resp, attr)
                if isinstance(v, str):
                    return v
                if isinstance(v, dict) and isinstance(v.get("content"), str):
                    return v["content"]
    except Exception:
        pass
    try:
        return json.dumps(resp, default=lambda o: getattr(o, "__dict__", str(o)), indent=2)
    except Exception:
        return str(resp)

def record_to_wav(target_path: str, duration: int = 5, samplerate: int = 16000) -> Tuple[bool, Optional[str]]:
    if sd is None or sf is None:
        return False, "sounddevice/soundfile not installed"
    try:
        data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()
        sf.write(target_path, data, samplerate)
        return True, None
    except Exception:
        return False, tbmod.format_exc()

def transcribe_with_whisper_model(model_obj, wav_path: str, language: Optional[str] = None) -> Tuple[str, Optional[str]]:
    if model_obj is None:
        return "", "Whisper model not loaded"
    try:
        kwargs = {}
        if language:
            kwargs["language"] = language
        r = model_obj.transcribe(wav_path, **kwargs)
        return r.get("text", "").strip(), None
    except Exception:
        return "", tbmod.format_exc()

def transcribe_with_speech_recognition(wav_path: str, language: str = "en-US") -> Tuple[str, Optional[str]]:
    if sr is None:
        return "", "SpeechRecognition not installed"
    r = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            audio = r.record(source)
        text = r.recognize_google(audio, language=language)
        return text, None
    except Exception:
        return "", tbmod.format_exc()

async def list_edge_voices(limit: int = 80) -> List[Dict[str, Any]]:
    if edge_tts is None:
        return []
    allv = await edge_tts.list_voices()
    return allv[:limit]

def list_pyttsx3_voices() -> List[Dict[str, str]]:
    if pyttsx3 is None:
        return []
    engine = pyttsx3.init()
    vs = engine.getProperty("voices")
    out = []
    for v in vs:
        out.append({"id": getattr(v, "id", None), "name": getattr(v, "name", None)})
    return out

def pyttsx3_speak_with_voice(text: str, voice_id: Optional[str] = None) -> bool:
    if pyttsx3 is None:
        return False
    try:
        engine = pyttsx3.init()
        if voice_id is not None:
            try:
                engine.setProperty("voice", voice_id)
            except Exception:
                pass
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception:
        return False

def _open_file_with_default_app(path: str) -> None:
    system = platform.system()
    try:
        if system == "Windows":
            os.startfile(path)
        elif system == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass

def _trim_message_content(msg: Dict[str, Any]) -> Dict[str, Any]:
    c = msg.get("content", "")
    if not isinstance(c, str):
        return msg
    if len(c) > MAX_CHARS_PER_MESSAGE:
        new = dict(msg)
        new["content"] = c[-MAX_CHARS_PER_MESSAGE:]
        return new
    return msg

def build_messages_for_model(full_messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    recent = full_messages[-HISTORY_MAX_MESSAGES:]
    out: List[Dict[str, str]] = []
    for m in recent:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not content:
            continue
        if len(content) > MAX_CHARS_PER_MESSAGE:
            content = content[-MAX_CHARS_PER_MESSAGE:]
        out.append({"role": role, "content": content})
    return out

CONV_DIR = Path(__file__).parent / "conversations"
CONV_DIR.mkdir(parents=True, exist_ok=True)

def _safe_title_to_filename(title: str) -> str:
    t = title.strip() or "untitled"
    t = re.sub(r"[^\w\s-]", "", t)
    t = re.sub(r"\s+", "_", t)
    if not t:
        t = "untitled"
    return t[:80]

def conversation_default_structure(title: str = "New Chat") -> Dict[str, Any]:
    return {
        "id": uuid.uuid4().hex,
        "title": title,
        "created_at": int(time.time()),
        "model": None,
        "messages": []  }

def save_conversation_to_file(conv: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Save a conversation JSON. Trim very old messages to SAVE_MAX_MESSAGES to avoid huge files.
    """
    try:
        msgs = conv.get("messages", []) or []
        if len(msgs) > SAVE_MAX_MESSAGES:
            truncated_count = len(msgs) - SAVE_MAX_MESSAGES
            new_msgs = msgs[-SAVE_MAX_MESSAGES:]
            note = {
                "role": "system",
                "content": f"(conversation truncated: removed {truncated_count} older messages to save disk space)",
                "ts": int(time.time()),
            }
            new_msgs.insert(0, note)
            conv["messages"] = new_msgs
        outpath: Optional[Path] = None
        cid = conv.get("id")
        if cid:
            for p in CONV_DIR.iterdir():
                if cid in p.name and p.suffix.lower() == ".json":
                    outpath = p
                    break
        if outpath is None:
            fn = _safe_title_to_filename(conv.get("title", "chat"))
            fname = f"{fn}_{conv.get('id')}.json"
            outpath = CONV_DIR / fname
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(conv, f, indent=2, ensure_ascii=False)
        return True, None
    except Exception:
        return False, tbmod.format_exc()


def load_conversation_from_file(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            conv = json.load(f)
            return conv, None
    except Exception:
        return None, tbmod.format_exc()


def list_conversation_files() -> List[Path]:
    return sorted([p for p in CONV_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".json"], key=lambda p: p.stat().st_mtime, reverse=True)


class OllamaWorker(QtCore.QThread):
    finished_signal = QtCore.Signal(object, object)  

    def __init__(self, client, model: Optional[str], messages: List[Dict[str, str]], max_tokens=MAX_TOKENS_DEFAULT, temperature=0.2, parent=None):
        super().__init__(parent)
        self.client = client
        self.model = model
        self.messages = messages
        self.max_tokens = max_tokens
        self.temperature = temperature

    def run(self):
        try:
            if self.isInterruptionRequested():
                self.finished_signal.emit("", "interrupted")
                return
            if self.client and self.model:
                resp = call_ollama_chat(self.client, self.model, self.messages, max_tokens=self.max_tokens, temperature=self.temperature)
                reply = extract_text_from_response(resp).strip()
            else:
                latest_user = ""
                for m in reversed(self.messages):
                    if m.get("role") == "user" and m.get("content"):
                        latest_user = m.get("content")
                        break
                if latest_user:
                    reply = "[echo] " + latest_user
                else:
                    reply = "[echo] (no user content)"
            if not reply:
                reply = "(no response)"
            self.finished_signal.emit(reply, None)
        except Exception:
            self.finished_signal.emit("", tbmod.format_exc())

class RecordTranscribeWorker(QtCore.QThread):
    finished_signal = QtCore.Signal(object, object)  # text, error

    def __init__(self, duration: int = 5, use_whisper: bool = True, parent=None):
        super().__init__(parent)
        self.duration = duration
        self.use_whisper = use_whisper

    def run(self):
        outname = f"record_{int(time.time())}.wav"
        ok, err = record_to_wav(outname, duration=self.duration)
        if not ok:
            self.finished_signal.emit("", err)
            return
        transcribed = ""
        if self.use_whisper and whisper is not None:
            try:
                model = whisper.load_model("tiny")
                transcribed, terr = transcribe_with_whisper_model(model, outname, language=None)
                if terr:
                    transcribed = ""
            except Exception:
                transcribed = ""
        if not transcribed:
            ttext, terr = transcribe_with_speech_recognition(outname, language="en-US")
            if ttext:
                transcribed = ttext
            else:
                if terr:
                    self.finished_signal.emit("", terr)
                    return
                transcribed = ""
        self.finished_signal.emit(transcribed, None)

class TTSPlayWorker(QtCore.QThread):
    finished_signal = QtCore.Signal(bool, object)  # ok, error

    def __init__(self, text: str, backend: str, voice_id: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.text = text
        self.backend = backend
        self.voice_id = voice_id

    def run(self):
        try:
            if self.backend == "pyttsx3":
                ok = pyttsx3_speak_with_voice(self.text, voice_id=self.voice_id)
                if not ok:
                    self.finished_signal.emit(False, "pyttsx3 failed or not installed")
                    return
                self.finished_signal.emit(True, None)
            elif self.backend == "edge":
                if edge_tts is None:
                    self.finished_signal.emit(False, "edge-tts not installed")
                    return
                mp3name = f"tts_{uuid.uuid4().hex}.mp3"
                try:
                    import asyncio
                    async def _save():
                        communicator = edge_tts.Communicate(self.text, self.voice_id)
                        await communicator.save(mp3name)
                    asyncio.run(_save())
                    _open_file_with_default_app(mp3name)
                    self.finished_signal.emit(True, None)
                except Exception:
                    self.finished_signal.emit(False, tbmod.format_exc())
            else:
                self.finished_signal.emit(False, f"Unknown TTS backend: {self.backend}")
        except Exception:
            self.finished_signal.emit(False, tbmod.format_exc())

class InitOllamaThread(QtCore.QThread):
    finished_signal = QtCore.Signal(object, object, object)  # client, model, err

    def __init__(self, parent=None):
        super().__init__(parent)

    def run(self):
        try:
            client = None
            model = None
            if ollama is not None:
                client = make_ollama_client()
                model = choose_model_for_client(client, requested=None)
            self.finished_signal.emit(client, model, None)
        except Exception:
            self.finished_signal.emit(None, None, tbmod.format_exc())

class EdgeVoicesLoader(QtCore.QThread):
    finished_signal = QtCore.Signal(object, object)  # voices, err

    def __init__(self, limit=200, parent=None):
        super().__init__(parent)
        self.limit = limit

    def run(self):
        if edge_tts is None:
            self.finished_signal.emit([], "edge-tts not installed")
            return
        try:
            import asyncio
            voices = asyncio.run(list_edge_voices(limit=self.limit))
            self.finished_signal.emit(voices, None)
        except Exception:
            self.finished_signal.emit([], tbmod.format_exc())

class ChatWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Merged Chatbot GUI — Multi-chat (history, low-resource)")
        self.resize(1100, 700)

        
        self.init_thread: Optional[InitOllamaThread] = None
        self.record_worker: Optional[RecordTranscribeWorker] = None
        self.ollama_worker: Optional[OllamaWorker] = None
        self.tts_worker: Optional[TTSPlayWorker] = None
        self.edge_loader: Optional[EdgeVoicesLoader] = None

        self.ollama_client = None
        self.ollama_model = None
        self.selected_tts_backend = ""   
        self.selected_voice = None       

        self.current_conversation: Optional[Dict[str, Any]] = None
        self.current_conv_path: Optional[Path] = None

        self._build_ui()
        self._init_ollama()
        self.load_conversations_list()

    def _build_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        left_w = QtWidgets.QWidget()
        left_l = QtWidgets.QVBoxLayout(left_w)
        left_l.setContentsMargins(4, 4, 4, 4)

        lbl = QtWidgets.QLabel("Conversations")
        left_l.addWidget(lbl)

        self.conv_list = QtWidgets.QListWidget()
        self.conv_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.conv_list.itemSelectionChanged.connect(self.on_conv_selection_changed)
        left_l.addWidget(self.conv_list, 1)

        conv_btns = QtWidgets.QHBoxLayout()
        self.new_conv_btn = QtWidgets.QPushButton("New")
        self.new_conv_btn.clicked.connect(self.create_new_conversation)
        conv_btns.addWidget(self.new_conv_btn)
        self.rename_conv_btn = QtWidgets.QPushButton("Rename")
        self.rename_conv_btn.clicked.connect(self.rename_selected_conversation)
        conv_btns.addWidget(self.rename_conv_btn)
        self.delete_conv_btn = QtWidgets.QPushButton("Delete")
        self.delete_conv_btn.clicked.connect(self.delete_selected_conversation)
        conv_btns.addWidget(self.delete_conv_btn)
        left_l.addLayout(conv_btns)

        main_layout.addWidget(left_w, 1)

        right_w = QtWidgets.QWidget()
        right_l = QtWidgets.QVBoxLayout(right_w)
        right_l.setContentsMargins(6, 6, 6, 6)

        self.status_label = QtWidgets.QLabel("Initializing...")
        self.status_label.setWordWrap(True)
        right_l.addWidget(self.status_label)

        input_group = QtWidgets.QGroupBox("Input")
        il = QtWidgets.QVBoxLayout(input_group)
        self.input_text = QtWidgets.QPlainTextEdit()
        self.input_text.setPlaceholderText("Type your question here (or use Record)...")
        il.addWidget(self.input_text)

        hrec = QtWidgets.QHBoxLayout()
        self.record_btn = QtWidgets.QPushButton("Record")
        self.record_btn.clicked.connect(self.on_record_clicked)
        hrec.addWidget(self.record_btn)

        self.duration_spin = QtWidgets.QSpinBox()
        self.duration_spin.setRange(1, 60)
        self.duration_spin.setValue(5)
        self.duration_spin.setSuffix(" s")
        hrec.addWidget(QtWidgets.QLabel("Duration:"))
        hrec.addWidget(self.duration_spin)

        self.transcribe_label = QtWidgets.QLabel("")
        hrec.addWidget(self.transcribe_label)
        hrec.addStretch()
        il.addLayout(hrec)
        right_l.addWidget(input_group)

        tts_group = QtWidgets.QGroupBox("Voice output")
        tl = QtWidgets.QHBoxLayout(tts_group)
        self.use_tts_cb = QtWidgets.QCheckBox("Enable voice output")
        tl.addWidget(self.use_tts_cb)
        self.choose_voice_btn = QtWidgets.QPushButton("Choose voice...")
        self.choose_voice_btn.clicked.connect(self.on_choose_voice)
        tl.addWidget(self.choose_voice_btn)
        self.chosen_voice_label = QtWidgets.QLabel("No voice chosen")
        tl.addWidget(self.chosen_voice_label)
        tl.addStretch(1)
        right_l.addWidget(tts_group)

        # Actions
        actions = QtWidgets.QHBoxLayout()
        self.send_btn = QtWidgets.QPushButton("Send")
        self.send_btn.clicked.connect(self.on_send_clicked)
        actions.addWidget(self.send_btn)

        self.play_reply_btn = QtWidgets.QPushButton("Play last reply")
        self.play_reply_btn.setEnabled(False)
        self.play_reply_btn.clicked.connect(self.on_play_reply)
        actions.addWidget(self.play_reply_btn)

        actions.addStretch(1)
        right_l.addLayout(actions)

        reply_group = QtWidgets.QGroupBox("Conversation")
        rl = QtWidgets.QVBoxLayout(reply_group)
        self.reply_text = QtWidgets.QPlainTextEdit()
        self.reply_text.setReadOnly(True)
        rl.addWidget(self.reply_text)
        right_l.addWidget(reply_group, 1)

        bottom = QtWidgets.QHBoxLayout()
        self.model_label = QtWidgets.QLabel("Ollama: not initialized")
        bottom.addWidget(self.model_label)
        bottom.addStretch(1)
        right_l.addLayout(bottom)

        main_layout.addWidget(right_w, 3)

    def _init_ollama(self):
        self.status_label.setText("Detecting Ollama client and models...")
        self.init_thread = InitOllamaThread(parent=self)
        self.init_thread.finished_signal.connect(self._on_ollama_init_done)
        self.init_thread.start()

    @QtCore.Slot(object, object, object)
    def _on_ollama_init_done(self, client, model, err):
        self.ollama_client = client
        self.ollama_model = model
        if client and model:
            self.status_label.setText("Ollama ready.")
            self.model_label.setText(f"Default Ollama model: {model}")
        elif client and not model:
            self.status_label.setText("Ollama client present but no model auto-selected.")
            self.model_label.setText("Ollama: client present, no model selected")
        else:
            self.status_label.setText("Ollama not available; replies will be echoed.")
            self.model_label.setText("Ollama: unavailable")
        self.init_thread = None

    def load_conversations_list(self):
        """Populate the conversation list widget from files on disk"""
        self.conv_list.clear()
        files = list_conversation_files()
        if not files:
            conv = conversation_default_structure("Welcome")
            save_conversation_to_file(conv)
            files = list_conversation_files()

        for p in files:
            conv, err = load_conversation_from_file(p)
            if conv:
                item = QtWidgets.QListWidgetItem(conv.get("title", p.stem))
                item.setData(QtCore.Qt.UserRole, str(p))
                self.conv_list.addItem(item)

        if self.conv_list.count() > 0:
            self.conv_list.setCurrentRow(0)

    def create_new_conversation(self):
        title, ok = QtWidgets.QInputDialog.getText(self, "New conversation", "Title:", QtWidgets.QLineEdit.Normal, "New Chat")
        if not ok:
            return
        conv = conversation_default_structure(str(title))
        ok, err = save_conversation_to_file(conv)
        if not ok:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to create conversation:\n{err}")
            return
        self.load_conversations_list()
        for i in range(self.conv_list.count()):
            it = self.conv_list.item(i)
            path = Path(it.data(QtCore.Qt.UserRole))
            if conv.get("id") in path.name:
                self.conv_list.setCurrentRow(i)
                break

    def rename_selected_conversation(self):
        it = self.conv_list.currentItem()
        if not it:
            QtWidgets.QMessageBox.information(self, "No selection", "Select a conversation first.")
            return
        path = Path(it.data(QtCore.Qt.UserRole))
        conv, err = load_conversation_from_file(path)
        if conv is None:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load conversation for rename:\n{err}")
            return
        new_title, ok = QtWidgets.QInputDialog.getText(self, "Rename conversation", "New title:", QtWidgets.QLineEdit.Normal, conv.get("title", ""))
        if not ok:
            return
        conv["title"] = str(new_title)
        ok, err = save_conversation_to_file(conv)
        if not ok:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to save conversation:\n{err}")
            return
        self.load_conversations_list()
        for i in range(self.conv_list.count()):
            it = self.conv_list.item(i)
            path = Path(it.data(QtCore.Qt.UserRole))
            if conv.get("id") in path.name:
                self.conv_list.setCurrentRow(i)
                break

    def delete_selected_conversation(self):
        it = self.conv_list.currentItem()
        if not it:
            QtWidgets.QMessageBox.information(self, "No selection", "Select a conversation first.")
            return
        path = Path(it.data(QtCore.Qt.UserRole))
        reply = QtWidgets.QMessageBox.question(self, "Delete conversation", f"Delete '{it.text()}'?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if reply != QtWidgets.QMessageBox.Yes:
            return
        try:
            path.unlink()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to delete file:\n{tbmod.format_exc()}")
            return
        self.load_conversations_list()

    def on_conv_selection_changed(self):
        it = self.conv_list.currentItem()
        if not it:
            self.current_conversation = None
            self.current_conv_path = None
            self.reply_text.clear()
            return
        path = Path(it.data(QtCore.Qt.UserRole))
        conv, err = load_conversation_from_file(path)
        if conv is None:
            self.status_label.setText("Failed to load conversation")
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load conversation:\n{err}")
            return
        if self.current_conversation:
            save_conversation_to_file(self.current_conversation)
        self.current_conversation = conv
        self.current_conv_path = path
        self.update_conversation_display()

    def update_conversation_display(self):
        """Render the conversation into the reply_text widget, but limit the amount shown."""
        if not self.current_conversation:
            self.reply_text.clear()
            return
        items = []
        msgs = self.current_conversation.get("messages", []) or []
        for m in msgs[-UI_MAX_MESSAGES_DISPLAY:]:
            role = m.get("role", "user")
            ts = m.get("ts")
            tstr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else ""
            prefix = "You" if role == "user" else ("Assistant" if role == "assistant" else role)
            content = m.get('content','')
            if isinstance(content, str) and len(content) > 20000:
                content = content[-20000:]
            items.append(f"{prefix} [{tstr}]:\n{content}\n")
        self.reply_text.setPlainText("\n".join(items))
        title = self.current_conversation.get("title", "Chat")
        conv_model = self.current_conversation.get("model")
        if conv_model:
            self.model_label.setText(f"Conv model: {conv_model} (default: {self.ollama_model})")
        else:
            self.model_label.setText(f"Default Ollama model: {self.ollama_model}")
        self.setWindowTitle(f"Merged Chatbot GUI — {title}")

    def on_record_clicked(self):
        if sd is None or sf is None:
            QtWidgets.QMessageBox.warning(self, "Recording not available", "sounddevice/soundfile not installed.")
            return
        duration = int(self.duration_spin.value())
        self.record_btn.setEnabled(False)
        self.transcribe_label.setText("Recording...")
        self.record_worker = RecordTranscribeWorker(duration=duration, use_whisper=(whisper is not None), parent=self)
        self.record_worker.finished_signal.connect(self._on_record_finished)
        self.record_worker.start()

    @QtCore.Slot(object, object)
    def _on_record_finished(self, text: str, error):
        self.record_btn.setEnabled(True)
        if error:
            self.transcribe_label.setText("Error")
            self.status_label.setText(f"Transcription error: {error}")
            self.record_worker = None
            return
        self.transcribe_label.setText("Transcribed")
        self.input_text.setPlainText(text)
        self.status_label.setText("Transcription complete.")
        self.record_worker = None

    def on_choose_voice(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Choose TTS voice")
        dialog.resize(800, 450)
        vlayout = QtWidgets.QVBoxLayout(dialog)

        tabs = QtWidgets.QTabWidget()
        vlayout.addWidget(tabs)
        edge_widget = QtWidgets.QWidget()
        el = QtWidgets.QVBoxLayout(edge_widget)
        self.edge_list = QtWidgets.QListWidget()
        el.addWidget(self.edge_list)
        edge_btns = QtWidgets.QHBoxLayout()
        edge_load_btn = QtWidgets.QPushButton("Load Edge voices")
        edge_btns.addWidget(edge_load_btn)
        edge_btns.addStretch()
        el.addLayout(edge_btns)
        tabs.addTab(edge_widget, "Edge TTS")
        py_widget = QtWidgets.QWidget()
        pl = QtWidgets.QVBoxLayout(py_widget)
        self.py_list = QtWidgets.QListWidget()
        pl.addWidget(self.py_list)
        py_btns = QtWidgets.QHBoxLayout()
        py_load_btn = QtWidgets.QPushButton("Load local voices")
        py_btns.addWidget(py_load_btn)
        py_btns.addStretch()
        pl.addLayout(py_btns)
        tabs.addTab(py_widget, "Local (pyttsx3)")

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        vlayout.addWidget(btns)

        def on_load_edge():
            edge_load_btn.setEnabled(False)
            self.edge_list.clear()
            if edge_tts is None:
                QtWidgets.QMessageBox.warning(dialog, "edge-tts missing", "edge-tts not installed.")
                edge_load_btn.setEnabled(True)
                return
            self.edge_loader = EdgeVoicesLoader(limit=200, parent=self)
            def on_loaded(voices, err):
                edge_load_btn.setEnabled(True)
                if err:
                    QtWidgets.QMessageBox.warning(dialog, "Error", f"Failed to load Edge voices:\n{err}")
                else:
                    if not voices:
                        QtWidgets.QMessageBox.information(dialog, "No voices", "No Edge voices returned.")
                    for v in voices:
                        name = v.get("DisplayName") or v.get("Name") or v.get("ShortName") or str(v)
                        item = QtWidgets.QListWidgetItem(f"{name}  ({v.get('ShortName')})  {v.get('Locale')}")
                        item.setData(QtCore.Qt.UserRole, ("edge", v.get("ShortName")))
                        self.edge_list.addItem(item)
                self.edge_loader = None
            self.edge_loader.finished_signal.connect(on_loaded)
            try:
                self.edge_loader.start()
            except Exception as e:
                edge_load_btn.setEnabled(True)
                QtWidgets.QMessageBox.warning(dialog, "Error", f"Failed to start Edge loader thread:\n{tbmod.format_exc()}")
                self.edge_loader = None

        def on_load_py():
            py_load_btn.setEnabled(False)
            self.py_list.clear()
            if pyttsx3 is None:
                QtWidgets.QMessageBox.warning(dialog, "pyttsx3 missing", "pyttsx3 not installed.")
                py_load_btn.setEnabled(True)
                return
            try:
                vs = list_pyttsx3_voices()
                if not vs:
                    QtWidgets.QMessageBox.information(dialog, "No voices", "No local pyttsx3 voices found.")
                for v in vs:
                    item = QtWidgets.QListWidgetItem(f"{v.get('name')}")
                    item.setData(QtCore.Qt.UserRole, ("pyttsx3", v.get("id")))
                    self.py_list.addItem(item)
            except Exception:
                QtWidgets.QMessageBox.warning(dialog, "Error", f"Failed to list local voices: {tbmod.format_exc()}")
            py_load_btn.setEnabled(True)

        edge_load_btn.clicked.connect(on_load_edge)
        py_load_btn.clicked.connect(on_load_py)

        def on_accept():
            it = self.edge_list.currentItem() or self.py_list.currentItem()
            if not it:
                QtWidgets.QMessageBox.information(dialog, "No selection", "No voice selected.")
                return
            backend, vid = it.data(QtCore.Qt.UserRole)
            self.selected_tts_backend = backend
            self.selected_voice = vid
            self.chosen_voice_label.setText(f"{backend}: {str(vid)[:120]}")
            dialog.accept()

        btns.accepted.connect(on_accept)
        btns.rejected.connect(dialog.reject)

        dialog.exec()
    def on_send_clicked(self):
        user_text = self.input_text.toPlainText().strip()
        if not user_text:
            QtWidgets.QMessageBox.information(self, "No input", "Type a question or record audio first.")
            return
        if not self.current_conversation:
            QtWidgets.QMessageBox.information(self, "No conversation", "Create or select a conversation first.")
            return
        msg = {"role": "user", "content": user_text, "ts": int(time.time())}
        self.current_conversation.setdefault("messages", []).append(msg)
        ok, err = save_conversation_to_file(self.current_conversation)
        if not ok:
            self.status_label.setText("Failed to save user message.")
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to save conversation:\n{err}")
        self.update_conversation_display()
        self.input_text.clear()
        messages_for_model = build_messages_for_model(self.current_conversation.get("messages", []))
        conv_model = self.current_conversation.get("model") or self.ollama_model
        self.send_btn.setEnabled(False)
        self.status_label.setText("Sending conversation history to model (trimmed)...")
        self.ollama_worker = OllamaWorker(self.ollama_client, conv_model, messages_for_model, max_tokens=MAX_TOKENS_DEFAULT, temperature=0.2, parent=self)
        self.ollama_worker.finished_signal.connect(self._on_ollama_reply)
        self.ollama_worker.start()

    @QtCore.Slot(object, object)
    def _on_ollama_reply(self, reply: str, error):
        self.send_btn.setEnabled(True)
        if error:
            self.status_label.setText("Chat error: saved an error note in conversation.")
            note = f"(model error)\n{error}"
            msg = {"role": "assistant", "content": note, "ts": int(time.time())}
            if self.current_conversation:
                self.current_conversation.setdefault("messages", []).append(msg)
                save_conversation_to_file(self.current_conversation)
                self.update_conversation_display()
            QtWidgets.QMessageBox.warning(self, "Chat error", f"Model call failed:\n{error}")
            self.ollama_worker = None
            return
        reply_text = reply or "(no reply)"
        msg = {"role": "assistant", "content": reply_text, "ts": int(time.time())}
        if self.current_conversation is None:
            self.status_label.setText("No conversation to store reply.")
            return
        self.current_conversation.setdefault("messages", []).append(msg)
        ok, serr = save_conversation_to_file(self.current_conversation)
        if not ok:
            self.status_label.setText("Saved reply failed.")
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to save conversation:\n{serr}")
        else:
            self.status_label.setText("Reply ready.")
        self.update_conversation_display()
        self.play_reply_btn.setEnabled(True)
        if self.use_tts_cb.isChecked() and self.selected_tts_backend:
            self._play_tts_for_text(reply_text)
        self.ollama_worker = None
    def on_play_reply(self):
        last_assistant = ""
        if self.current_conversation:
            for m in reversed(self.current_conversation.get("messages", [])):
                if m.get("role") == "assistant":
                    last_assistant = m.get("content", "")
                    break
        if not last_assistant:
            QtWidgets.QMessageBox.information(self, "No assistant reply", "No assistant reply to play.")
            return
        if not self.selected_tts_backend:
            QtWidgets.QMessageBox.information(self, "No voice", "Choose a voice first.")
            return
        self._play_tts_for_text(last_assistant)
    def _play_tts_for_text(self, text: str):
        self.play_reply_btn.setEnabled(False)
        self.status_label.setText("Speaking...")
        self.tts_worker = TTSPlayWorker(text, backend=self.selected_tts_backend, voice_id=self.selected_voice, parent=self)
        self.tts_worker.finished_signal.connect(self._on_tts_finished)
        self.tts_worker.start()
    @QtCore.Slot(bool, object)
    def _on_tts_finished(self, ok: bool, err):
        self.play_reply_btn.setEnabled(True)
        if ok:
            self.status_label.setText("TTS finished.")
        else:
            self.status_label.setText("TTS error.")
            if err:
                QtWidgets.QMessageBox.warning(self, "TTS error", str(err))
        self.tts_worker = None
    def closeEvent(self, event: QtGui.QCloseEvent):
        self.status_label.setText("Closing — waiting for background tasks to stop...")
        threads = [self.record_worker, self.ollama_worker, self.tts_worker, self.edge_loader, self.init_thread]
        for t in threads:
            if t is not None and isinstance(t, QtCore.QThread):
                try:
                    t.requestInterruption()
                except Exception:
                    pass
                try:
                    t.quit()
                except Exception:
                    pass
        wait_deadline = time.time() + 5.0
        for t in threads:
            if t is not None and isinstance(t, QtCore.QThread):
                remaining = max(0, int((wait_deadline - time.time()) * 1000))
                try:
                    t.wait(remaining)
                except Exception:
                    pass
        event.accept()
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = ChatWindow()
    win.show()
    sys.exit(app.exec())
if __name__ == "__main__":
    main()
