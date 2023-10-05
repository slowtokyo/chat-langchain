"""Main entrypoint for the app."""
import logging
import chromadb
import json
from typing import Optional
from dotenv import load_dotenv

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from langchain.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from langchain.vectorstores import VectorStore, Chroma

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse
from dxr_loader import redact_text
import os
from ingest import ingest_docs

os.environ["TOKENIZERS_PARALLELISM"] = "false"
DXR_API_KEY = os.environ["DXR_API_KEY"]


load_dotenv()

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:9000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None


@app.on_event("startup")
async def startup_event():
    logging.info("loading vectorstore")
    global vectorstore
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    persist_directory = "db"
    collection_name = "dxr_unredacted"
    persistent_client = chromadb.PersistentClient(persist_directory)
    vectorstore = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    print("There are", vectorstore._collection.count(), "in the collection")


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# @app.post("/load")
# async def loadData(request: Request):
#     dxr_label = Request.body["dxr_label"]
#     print(dxr_label)
#     ingest_docs(dxr_label)
#     return "loading data into DB..."


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_chain(vectorstore, question_handler, stream_handler)
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # Receive and send back the client message
            message = await websocket.receive_text()
            message = json.loads(message)
            question = message["question"]
            redact = message["redact"]
            if redact:
                question = redact_text(
                    "https://demo.dataxray.io/api",
                    question,
                    [21],
                    DXR_API_KEY,
                )
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.model_dump())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.model_dump())

            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history}
            )
            chat_history.append((question, result["answer"]))

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.model_dump())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.model_dump())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
