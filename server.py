from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BertTokenizerFast, BertForQuestionAnswering
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

# Khởi tạo FastAPI
app = FastAPI()

# Load dataset
with open("dataset.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

# Load model QA
model = BertForQuestionAnswering.from_pretrained("yenly1234/chatbot")
tokenizer = BertTokenizerFast.from_pretrained("yenly1234/chatbot")

# Load model embedding
embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # Rất nhẹ và tốt
# Encode toàn bộ câu hỏi trong dataset
questions = [item["question"] for item in test_data]
question_embeddings = embed_model.encode(questions, convert_to_numpy=True)

# Build FAISS index
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)

# Map index -> data
id_to_sample = {i: item for i, item in enumerate(test_data)}

# Định nghĩa schema request
class QuestionRequest(BaseModel):
    question: str

# Hàm hỏi đáp
def chat(question):
    query_vec = embed_model.encode([question], convert_to_numpy=True)
    D, I = index.search(query_vec, 1)  # top-1 match

    idx = int(I[0][0])
    sample = id_to_sample[idx]
    context = sample["answer"]

    # Chuẩn bị inputs cho model QA
    inputs = tokenizer(question, context, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    start = torch.argmax(start_scores)
    end = torch.argmax(end_scores)

    if end < start:
        end = start

    answer_tokens = inputs["input_ids"][0][start:end + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return {
        "question":    sample["question"],
        "label":       sample["label"],
        "language":    sample["language"],
        "context":     sample["context"],
        "answer":      sample["answer"],
        "start_char":  sample["start_char"],
        "end_char":    sample["end_char"]
    }

# API endpoint
@app.post("/chat")
def chatbot_api(data: QuestionRequest):
    response = chat(data.question)
    return response

# Chạy server (nếu cần chạy trực tiếp)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)