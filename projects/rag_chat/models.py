from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    source: list[str]

class UploadResponse(BaseModel):
    message: str
    file_name: str
    chunks: int

class DeleteResponse(BaseModel):
    message: str