from langgraph.graph import StateGraph, MessagesState, START, END
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.message import add_messages
import sqlite3
from langchain_core.messages import SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import psycopg
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()
model = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")

DB_URI = "postgresql://postgres:1234@localhost:5432/postgres?sslmode=disable"
search_tool = DuckDuckGoSearchRun(region="us-en")

def db():
    return psycopg.connect(DB_URI, autocommit=True)

@tool
def add_note(content: str) -> dict:
    """
    Save a note for the user. Use this when the user asks to note, remember to write down something.
    """

    with db() as conn:
        row = conn.execute(
            "INSERT INTO notes (content) VALUES (%s) RETURNING id", (content,)
        ).fetchone()
    return f"Saved note #{row[0]}"

@tool
def list_notes() -> str:
    """ Show all saved notes. Use this when user asks to see, read or list theire notes."""
    with db() as conn:
        rows = conn.execute(
            "SELECT id, content FROM notes ORDER BY id"
        ).fetchone()
    if not rows:
        return "No notes saved yet."
    return "\n".join(f"#{i}: {c}" for i, c in rows)

@tool
def delate_note(note_id: int) -> str:
    """Delate a note by its id number."""
    with db() as conn:
        cur = conn.execute("DELATE FROMnotes WHERE id = %s", (note_id,))
    return f"Delated note#{note_id}" if cur.rowcount else f"no note with id {note_id}"

tools = [search_tool, add_note, list_notes, delate_note]

model_with_tools = model.bind_tools(tools)

class bot(TypedDict):
    messages: Annotated[list, add_messages]
   
def chatbot(state: MessagesState):

    system = SystemMessage(content='You are a chatbot like alexa and you always talk in angry tone to users')
    response = model_with_tools.invoke([system] + state["messages"])

    return {'messages' : [response]}


builder = StateGraph(MessagesState)

builder.add_node("chatbot", chatbot)
builder.add_node('tools', ToolNode(tools))

builder.add_edge(START, 'chatbot')
builder.add_conditional_edges('chatbot', tools_condition)
builder.add_edge('chatbot', END)


with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()

    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable" : {"thread_id":"thread-2"}}

    result = graph.invoke({"messages": [("user", "tell me what is today's biggest news")]}, config)

    print(result["messages"][-1].text)


# with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
#     checkpointer.setup()

#     graph = builder.compile(checkpointer=checkpointer)

# print(result["messages"][-3].text)