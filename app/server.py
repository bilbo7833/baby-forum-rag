import logging
from fastapi import FastAPI, HTTPException
import uvicorn
from forum_rag import ForumRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()
rag = ForumRAG()


@app.get("/input")
def input(input_string: str):
    try:
        output = rag.input(input_string)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
