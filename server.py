# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import torch
# from transformers import BertTokenizerFast, BertForQuestionAnswering
# from sentence_transformers import SentenceTransformer
# import faiss
# import json
# import numpy as np

# # Khởi tạo FastAPI
# app = FastAPI()

# # Load dataset
# with open("dataset_with_key_answer (1).json", "r", encoding="utf-8") as f:
#     test_data = json.load(f)

# # Load model QA
# model = BertForQuestionAnswering.from_pretrained("yenly1234/BERTQAEN")
# tokenizer = BertTokenizerFast.from_pretrained("yenly1234/BERTQAEN")

# # Load model embedding
# embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # Rất nhẹ và tốt
# # Encode toàn bộ câu hỏi trong dataset
# questions = [item["question"] for item in test_data]
# question_embeddings = embed_model.encode(questions, convert_to_numpy=True)

# # Build FAISS index
# dimension = question_embeddings.shape[1]
# index = faiss.IndexFlatL2(dimension)
# index.add(question_embeddings)

# # Map index -> data
# id_to_sample = {i: item for i, item in enumerate(test_data)}

# # Định nghĩa schema request
# class QuestionRequest(BaseModel):
#     question: str

# # Hàm hỏi đáp
# def chat(query):
#     query_vec = embed_model.encode(query, convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)

#     # 🔍 Tìm top-5 tương tự
#     D, I = index.search(query_vec, 5)

#     # 📌 Chọn best match dựa trên cosine similarity
#     best_idx = None
#     best_score = -1
#     for score, idx in zip(D[0], I[0]):
#         sim = np.dot(query_vec[0], question_embeddings[idx])
#         if sim > best_score:
#             best_score = sim
#             best_idx = idx

#     matched = id_to_sample[best_idx]

#     # 🔄 (Tùy chọn) In 5 câu tương tự
#     print("Top 5 candidates:")
#     for score, idx in zip(D[0], I[0]):
#         print(f" - {dataset[idx]['question']} | Score: {score:.4f}")

#     return {
#         "your_question": query,
#         "matched_question": matched["question"],
#         "context": matched.get("context", ""),
#         "answer": matched["answer"],
#         "label": matched.get("label", ""),
#         "language": matched.get("language", ""),
#         "start_char": matched.get("start_char", -1),
#         "end_char": matched.get("end_char", -1),
#     }


# # API endpoint
# @app.post("/chat")
# def chatbot_api(data: QuestionRequest):
#     response = chat(data.question)
#     return response

# # Chạy server (nếu cần chạy trực tiếp)
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizerFast, BertForQuestionAnswering
import faiss
import json
import numpy as np
import torch

# 🚀 Khởi tạo FastAPI
app = FastAPI()

# 📦 Load dataset
with open("dataset_with_key_answer (1).json", "r", encoding="utf-8") as f:
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
model = BertForQuestionAnswering.from_pretrained("yenly1234/BERTQAEN")
tokenizer = BertTokenizerFast.from_pretrained("yenly1234/BERTQAEN")

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
        context = matched.get("key_answer", matched["answer"])


    # Bước 2: QA từ context + query
    inputs = tokenizer(query, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits)
    if end < start:
        end = start

    answer = tokenizer.decode(inputs["input_ids"][0][start:end+1], skip_special_tokens=True)

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