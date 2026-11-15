import queue, threading, re
import numpy as np, sounddevice as sd, webrtcvad, psycopg
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from piper import PiperVoice
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, AIMessageChunk
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

# ---------- models ----------
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
stt = WhisperModel("base", device="cuda", compute_type="int8_float16")  # use device="cpu", "int8" if CUDA errors
tts = PiperVoice.load(r"F:\Projects\Chaty\en_US-lessac-medium.onnx")
vad = webrtcvad.Vad(2)

DB_URI = "postgresql://postgres:1234@localhost:5432/postgres?sslmode=disable"
SR, FRAME_MS = 16000, 30
FRAME = SR * FRAME_MS // 1000
audio_q = queue.Queue()

def db():
    return psycopg.connect(DB_URI, autocommit=True)

with db() as conn:
    conn.execute("CREATE TABLE IF NOT EXISTS notes (id SERIAL PRIMARY KEY, content TEXT)")

# ---------- tools ----------
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def add_note(content: str) -> str:
    """Save a note for the user. Use when the user asks to note, remember or write down something."""
    with db() as conn:
        row = conn.execute("INSERT INTO notes (content) VALUES (%s) RETURNING id", (content,)).fetchone()
    return f"Saved note #{row[0]}"

@tool
def list_notes() -> str:
    """Show all saved notes. Use when the user asks to see, read or list their notes."""
    with db() as conn:
        rows = conn.execute("SELECT id, content FROM notes ORDER BY id").fetchall()
    if not rows:
        return "No notes saved yet."
    return "\n".join(f"#{i}: {c}" for i, c in rows)

@tool
def delete_note(note_id: int) -> str:
    """Delete a note by its id number."""
    with db() as conn:
        cur = conn.execute("DELETE FROM notes WHERE id = %s", (note_id,))
    return f"Deleted note #{note_id}" if cur.rowcount else f"No note with id {note_id}"

tools = [search_tool, add_note, list_notes, delete_note]
model_with_tools = model.bind_tools(tools)

# ---------- graph ----------
SYSTEM = SystemMessage(content=(
    "You are a voice assistant like Alexa and you always talk in an angry tone. "
    "Reply in short, plain sentences. No markdown, lists, or emojis."
))

def chatbot(state: MessagesState):
    return {"messages": [model_with_tools.invoke([SYSTEM] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", tools_condition)  # goes to tools or END
builder.add_edge("tools", "chatbot")

# ---------- audio ----------
def listen():
    frames, silence, speech_frames, started = [], 0, 0, False
    with sd.RawInputStream(samplerate=SR, blocksize=FRAME, dtype="int16", channels=1) as mic:
        while True:
            data, _ = mic.read(FRAME)
            data = bytes(data)
            if vad.is_speech(data, SR):
                started, silence = True, 0
                speech_frames += 1
                frames.append(data)
            elif started:
                frames.append(data)
                silence += 1
                if silence > 20:            # ~600 ms silence = finished
                    if speech_frames < 8:   # too short, probably noise
                        frames, silence, speech_frames, started = [], 0, 0, False
                        continue
                    break
    return np.frombuffer(b"".join(frames), np.int16).astype(np.float32) / 32768

def transcribe(pcm):
    segs, _ = stt.transcribe(pcm, language="en", beam_size=1, vad_filter=False)
    return " ".join(s.text for s in segs).strip()

def tts_worker():
    out = sd.RawOutputStream(samplerate=tts.config.sample_rate, channels=1, dtype="int16")
    out.start()
    while True:
        sentence = audio_q.get()
        try:
            if hasattr(tts, "synthesize_stream_raw"):          # piper-tts 1.2.x
                for chunk in tts.synthesize_stream_raw(sentence):
                    out.write(chunk)
            else:                                              # piper-tts 1.3+
                for chunk in tts.synthesize(sentence):
                    out.write(chunk.audio_int16_bytes)
        finally:
            audio_q.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def clean(s):
    return re.sub(r"[*_#`]", "", s).strip()

SENT = re.compile(r"(.+?[.!?])\s")

def speak_reply(graph, user_text, config):
    buf = ""
    for chunk, meta in graph.stream(
        {"messages": [("user", user_text)]}, config, stream_mode="messages"
    ):
        if meta.get("langgraph_node") != "chatbot":
            continue
        if not isinstance(chunk, AIMessageChunk) or chunk.tool_call_chunks:
            continue
        buf += chunk.text
        while (m := SENT.search(buf)):
            audio_q.put(clean(m.group(1)))
            buf = buf[m.end():]
    if buf.strip():
        audio_q.put(clean(buf))
    audio_q.join()   # wait until finished speaking so the mic doesn't hear the bot

# ---------- main ----------
if __name__ == "__main__":
    with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
        checkpointer.setup()
        graph = builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "thread-2"}}

        print("Listening... (Ctrl+C to quit)")
        while True:
            text = transcribe(listen())
            if text:
                print("You:", text)
                speak_reply(graph, text, config)