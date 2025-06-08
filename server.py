
# from fastapi import FastAPI
# from pydantic import BaseModel
# from sentence_transformers import SentenceTransformer
# from transformers import BertTokenizerFast, BertForQuestionAnswering
# import faiss
# import json
# import numpy as np
# import torch

# # 🚀 Khởi tạo FastAPI
# app = FastAPI()

# # 📦 Load dataset
# with open("dataset_with_key_answer (1).json", "r", encoding="utf-8") as f:
#     dataset = json.load(f)

# # 🧠 Load embedding model (mạnh)
# embed_model = SentenceTransformer('all-mpnet-base-v2')

# # 🔢 Encode + normalize các câu hỏi trong dataset
# questions = [item["question"] for item in dataset]
# question_embeddings = embed_model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)

# # 🎯 FAISS index (cosine similarity = inner product vì đã normalize)
# index = faiss.IndexFlatIP(question_embeddings.shape[1])
# index.add(question_embeddings)
# id_to_sample = {i: item for i, item in enumerate(dataset)}

# # 🤖 Load model BERT QA
# model = BertForQuestionAnswering.from_pretrained("yenly1234/BERTQAEN")
# tokenizer = BertTokenizerFast.from_pretrained("yenly1234/BERTQAEN")

# # 📥 Schema request
# class QuestionRequest(BaseModel):
#     question: str

# # 💬 Hàm xử lý QA
# def chat(query):
#     # Bước 1: Encode và tìm nearest question
#     query_vec = embed_model.encode(query, convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)
#     D, I = index.search(query_vec, 5)

#     # Chọn câu hỏi giống nhất
#     best_idx = I[0][0]
#     matched = id_to_sample[best_idx]
#     if "```" in matched["answer"]:
#         context = matched["answer"]
#     else:
#         context = matched.get("key_answer", matched["answer"])


#     # Bước 2: QA từ context + query
#     inputs = tokenizer(query, context, return_tensors="pt", truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)

#     start = torch.argmax(outputs.start_logits)
#     end = torch.argmax(outputs.end_logits)
#     if end < start:
#         end = start

#     # answer = tokenizer.decode(inputs["input_ids"][0][start:end+1], skip_special_tokens=True)
#     answer_ids = inputs["input_ids"][0][start:end+len(context)]
#     answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

#     # Bước 3: Trả kết quả
#     return {
#         "your_question": query,
#         "matched_question": matched["question"],
#         "context_used": context,
#         "bert_generated_answer": answer,
#         "original_answer": matched["answer"],
#         "label": matched.get("label", ""),
#         "language": matched.get("language", ""),
#         "start_char": matched.get("start_char", -1),
#         "end_char": matched.get("end_char", -1),
#         "key_answer": matched.get("key_answer","")
#     }

# # 🌐 API endpoint
# @app.post("/chat")
# def chatbot_api(data: QuestionRequest):
#     return chat(data.question)

# # ▶️ Chạy server nếu gọi trực tiếp
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# from fastapi import FastAPI
# from pydantic import BaseModel
# import json
# import numpy as np
# import faiss
# import torch
# from sentence_transformers import SentenceTransformer
# from transformers import (
#     BertTokenizerFast,
#     BertForQuestionAnswering,
#     BartTokenizer,
#     BartForConditionalGeneration
# )

# # 🚀 Khởi tạo FastAPI
# app = FastAPI()

# # -------------------------------
# # 1) Load dataset
# # -------------------------------
# with open("dataset_with_key_answer (1).json", "r", encoding="utf-8") as f:
#     dataset = json.load(f)

# # -------------------------------
# # 2) Sentence-BERT + FAISS index for retrieval
# # -------------------------------
# embed_model = SentenceTransformer('all-mpnet-base-v2')
# questions = [item["question"] for item in dataset]
# q_embs = embed_model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
# index = faiss.IndexFlatIP(q_embs.shape[1])
# index.add(q_embs)
# id_to_sample = {i: item for i, item in enumerate(dataset)}

# # -------------------------------
# # 3) Load QA model (yenly1234/BERTQAEN)
# # -------------------------------
# qa_tokenizer = BertTokenizerFast.from_pretrained("yenly1234/BERTQAEN")
# qa_model = BertForQuestionAnswering.from_pretrained("yenly1234/BERTQAEN")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# qa_model.to(device)

# # -------------------------------
# # 4) Load BART for paraphrase
# # -------------------------------
# bart_tok = BartTokenizer.from_pretrained("facebook/bart-large")
# bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
# bart_model.to(device)

# def generate_with_bart(text: str, attempts: int = 5) -> str:
#     """
#     Generate paraphrase with BART: sample multiple candidates, using beam sampling for diversity.
#     """
#     inp = bart_tok(f"paraphrase: {text}", return_tensors="pt", truncation=True, max_length=256).to(device)
#     out = bart_model.generate(
#         **inp,
#         do_sample=True,
#         top_k=50,
#         top_p=0.95,
#         temperature=1.5,
#         num_return_sequences=attempts,
#         num_beams=attempts,
#         max_length=256,
#         no_repeat_ngram_size=3
#     )
#     candidates = [bart_tok.decode(o, skip_special_tokens=True).strip() for o in out]
#     def clean(p: str) -> str:
#         return p.replace("paraphrase:", "", 1).strip()
#     variants = [clean(p) for p in candidates]
#     for v in variants:
#         if v and v.lower() != text.lower():
#             return v
#     return variants[0] if variants else text

# # -------------------------------
# # 5) FastAPI chat logic
# # -------------------------------
# class QuestionRequest(BaseModel):
#     question: str

# @app.post("/chat")
# def chatbot_api(data: QuestionRequest):
#     query = data.question
#     # Step 1: semantic retrieval
#     q_vec = embed_model.encode(query, convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)
#     _, I = index.search(q_vec, 5)
#     best_idx = int(I[0][0])
#     matched = id_to_sample[best_idx]
#     context = matched.get("answer", "")

#     # If context contains code, return it directly
#     if '```' in context or context.strip().startswith(('def ', 'import ')):
#         return {
#             "your_question": query,
#             "matched_question": matched["question"],
#             "context_used": context,
#             "bert_generated_answer": context,
#             "original_answer": matched["answer"],
#             "label": matched.get("label", ""),
#             "language": matched.get("language", ""),
#             "start_char": matched.get("start_char", -1),
#             "end_char": matched.get("end_char", -1),
#             "key_answer": matched.get("key_answer",""),
#             "used_paraphrase": False
#         }
       

#     # Step 2: QA prediction
#     inputs = qa_tokenizer(query, context, return_tensors="pt", truncation=True, max_length=512).to(device)
#     with torch.no_grad():
#         outputs = qa_model(**inputs)
#     start = torch.argmax(outputs.start_logits)
#     end = torch.argmax(outputs.end_logits)
#     if end < start:
#         end = start
#     answer_ids = inputs.input_ids[0][start:end+1]
#     answer = qa_tokenizer.decode(answer_ids, skip_special_tokens=True)

#     # Step 3: Paraphrase with BART
#     final_para = generate_with_bart(answer)

#     return {
#         "your_question": query,
#         "matched_question": matched["question"],
#         "context_used": context,
#         "bert_generated_answer": final_para,
#         "original_answer": matched["answer"],
#         "label": matched.get("label", ""),
#         "language": matched.get("language", ""),
#         "start_char": matched.get("start_char", -1),
#         "end_char": matched.get("end_char", -1),
#         "key_answer": matched.get("key_answer",""),
#         "used_paraphrase": True
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
from fastapi import FastAPI
from pydantic import BaseModel
import json
import numpy as np
import faiss
import torch
import re
from sentence_transformers import SentenceTransformer
from transformers import (
    BertTokenizerFast,
    BertForQuestionAnswering,
    XLMRobertaTokenizerFast,
    XLMRobertaForQuestionAnswering,
    BartTokenizer,
    BartForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM
)
from langdetect import detect

# 🚀 Khởi tạo FastAPI
app = FastAPI()

# -------------------------------
# 1) Load dataset
# -------------------------------
with open("dataset_with_key_answer (1).json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# -------------------------------
# 2) Sentence-BERT + FAISS index for retrieval
# -------------------------------
embed_model = SentenceTransformer('all-mpnet-base-v2')
questions = [item["question"] for item in dataset]
q_embs = embed_model.encode(questions, convert_to_numpy=True, normalize_embeddings=True)
index = faiss.IndexFlatIP(q_embs.shape[1])
index.add(q_embs)
id_to_sample = {i: item for i, item in enumerate(dataset)}

# -------------------------------
# 3) Load QA models
# -------------------------------
vi_tokenizer = XLMRobertaTokenizerFast.from_pretrained("yenly1234/XMLBERTVI")
vi_model = XLMRobertaForQuestionAnswering.from_pretrained("yenly1234/XMLBERTVI").to("cuda" if torch.cuda.is_available() else "cpu")

en_tokenizer = BertTokenizerFast.from_pretrained("yenly1234/BERTQAEN")
en_model = BertForQuestionAnswering.from_pretrained("yenly1234/BERTQAEN").to("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 4) Load BART for paraphrase
# -------------------------------
bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")

bartpho_tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word")
bartpho_model = AutoModelForSeq2SeqLM.from_pretrained("vinai/bartpho-word")

def detect_language(text):
    try:
        lang = detect(text)
        return lang
    except:
        return "unknown"

def generate_paraphrase(text, max_length=256):
    lang = detect_language(text)

    if lang == "en":
        tokenizer = bart_tokenizer
        model = bart_model
        prefix = "paraphrase: "
    else:
        tokenizer = bartpho_tokenizer
        model = bartpho_model
        prefix = ""

    inputs = tokenizer(prefix + text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# -------------------------------
# 5) Helper: detect if text is Vietnamese
# -------------------------------
def is_vietnamese(text):
    return bool(re.search(r"[\u00C0-\u1EF9]", text))

# -------------------------------
# 6) FastAPI chat logic
# -------------------------------
class QuestionRequest(BaseModel):
    question: str

@app.post("/chat")
def chatbot_api(data: QuestionRequest):
    query = data.question

    # Detect language
    if is_vietnamese(query):
        qa_model, qa_tokenizer = vi_model, vi_tokenizer
        language = "vi"
    else:
        qa_model, qa_tokenizer = en_model, en_tokenizer
        language = "en"

    # Step 1: semantic retrieval
    q_vec = embed_model.encode(query, convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)
    _, I = index.search(q_vec, 5)
    best_idx = int(I[0][0])
    matched = id_to_sample[best_idx]
    context = matched.get("answer", "")

    if '```' in context or context.strip().startswith(('def ', 'import ')):
        return {
            "your_question": query,
            "matched_question": matched["question"],
            "context_used": context,
            "bert_generated_answer": context,
            "original_answer": matched["answer"],
            "label": matched.get("label", ""),
            "language": language,
            "start_char": matched.get("start_char", -1),
            "end_char": matched.get("end_char", -1),
            "key_answer": matched.get("key_answer",""),
            "used_paraphrase": False
        }

    # Step 2: QA prediction
    inputs = qa_tokenizer(query, context, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = qa_model(**inputs)
    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits)
    if end < start:
        end = start
    answer_ids = inputs.input_ids[0][start:end+1]
    answer = qa_tokenizer.decode(answer_ids, skip_special_tokens=True)

    # Step 3: Paraphrase with BART
    final_para = generate_paraphrase(answer)

    return {
        "your_question": query,
        "matched_question": matched["question"],
        "context_used": context,
        "bert_generated_answer": final_para,
        "original_answer": matched["answer"],
        "label": matched.get("label", ""),
        "language": language,
        "start_char": matched.get("start_char", -1),
        "end_char": matched.get("end_char", -1),
        "key_answer": matched.get("key_answer",""),
        "used_paraphrase": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


