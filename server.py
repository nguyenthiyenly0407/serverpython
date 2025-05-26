
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizerFast, BertForQuestionAnswering
import faiss
import json
import numpy as np
import torch
import random
# 🚀 Khởi tạo FastAPI
app = FastAPI()

# 📦 Load dataset
with open("dataset_with_key_answer.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# 🧠 Load embedding model (mạnh)
embed_model = SentenceTransformer('all-mpnet-base-v2')

# 🔢 Encode + normalize các câu hỏi trong dataset
questions = [item["question"] for item in dataset]
question_embeddings = embed_model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)

# 🎯 FAISS index (cosine similarity = inner product vì đã normalize)
index = faiss.IndexFlatIP(question_embeddings.shape[1])
index.add(question_embeddings)
id_to_sample = {i: item for i, item in enumerate(dataset)}

# 🤖 Load model BERT QA
model = BertForQuestionAnswering.from_pretrained("yenly1234/BERTQAENG")
tokenizer = BertTokenizerFast.from_pretrained("yenly1234/BERTQAENG")

# 📥 Schema request
class QuestionRequest(BaseModel):
    question: str

# 💬 Hàm xử lý QA
def chat(query):
    # Bước 1: Encode và tìm nearest question
    query_vec = embed_model.encode(query, convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)
    D, I = index.search(query_vec, 5)

    # Chọn câu hỏi giống nhất
    best_idx = I[0][0]
    matched = id_to_sample[best_idx]
    if "```" in matched["answer"]:
        context = matched["answer"]
    else:
       context = random.choice(matched["key_answer"])


    # Bước 2: QA từ context + query
    inputs = tokenizer(query, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits)
    if end < start:
        end = start

    # answer = tokenizer.decode(inputs["input_ids"][0][start:end+1], skip_special_tokens=True)
    answer_ids = inputs["input_ids"][0][start:end+len(context)]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

    # Bước 3: Trả kết quả
    return {
        "your_question": query,
        "matched_question": matched["question"],
        "context_used": context,
        "bert_generated_answer": answer,
        "original_answer": matched["answer"],
        "label": matched.get("label", ""),
        "language": matched.get("language", ""),
        "start_char": matched.get("start_char", -1),
        "end_char": matched.get("end_char", -1),
        "key_answer": matched.get("key_answer","")
    }

# 🌐 API endpoint
@app.post("/chat")
def chatbot_api(data: QuestionRequest):
    return chat(data.question)

# ▶️ Chạy server nếu gọi trực tiếp
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
