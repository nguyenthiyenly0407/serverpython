
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
from fastapi import FastAPI
from pydantic import BaseModel
import json
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoTokenizer,
    EncoderDecoderModel,
    BartTokenizer,
    BartForConditionalGeneration,
    BertTokenizerFast,
    BertForQuestionAnswering
)

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
# 3) Paraphrase pipeline: BERT2BERT primary + BART fallback
# -------------------------------
bert_ckpt = "yenly1234/BERTQAEN"
# Encoder-decoder model uses same tokenizer
bert_tokenizer = AutoTokenizer.from_pretrained(bert_ckpt)
bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained(bert_ckpt, bert_ckpt)
# configure special tokens
decoder_start = bert_tokenizer.bos_token_id or bert_tokenizer.cls_token_id
# Ensure both decoder_start_token_id and bos_token_id are set
bert2bert.config.decoder_start_token_id = decoder_start
bert2bert.config.bos_token_id = decoder_start
bert2bert.config.eos_token_id = bert_tokenizer.sep_token_id
bert2bert.config.pad_token_id = bert_tokenizer.pad_token_id
bert2bert.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

bart_tok = BartTokenizer.from_pretrained("facebook/bart-large")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
bart_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# generate with BART fallback
def generate_with_bart(text: str) -> str:
    inp = bart_tok(f"paraphrase: {text}", return_tensors="pt", truncation=True, max_length=256).to(bart_model.device)
    out = bart_model.generate(
        **inp,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
        num_return_sequences=1,
        max_length=256,
        no_repeat_ngram_size=3
    )
    para = bart_tok.decode(out[0], skip_special_tokens=True).strip()
    if para.lower().startswith("paraphrase:"):
        para = para.split(":",1)[1].strip()
    return para

# generate with BERT2BERT
def generate_paraphrase_with_bert2bert(text: str) -> str:
    enc = bert_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(bert2bert.device)
    out_ids = bert2bert.generate(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        decoder_start_token_id=bert2bert.config.decoder_start_token_id,
        bos_token_id=bert2bert.config.bos_token_id,
        eos_token_id=bert2bert.config.eos_token_id,
        pad_token_id=bert2bert.config.pad_token_id,
        max_length=128,
        num_beams=4,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        num_return_sequences=1
    )
    para = bert_tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    return para

# -------------------------------
# 4) FastAPI chat logic
# -------------------------------
class QuestionRequest(BaseModel):
    question: str

@app.post("/chat")
def chatbot_api(data: QuestionRequest):
    query = data.question
    # Retrieval
    q_vec = embed_model.encode(query, convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)
    D, I = index.search(q_vec, 5)
    best_idx = int(I[0][0])
    matched = id_to_sample[best_idx]
    base_answer = matched.get("answer", "")
    # Decide whether to paraphrase
    if '```' in base_answer or base_answer.strip().startswith('def ') or base_answer.strip().startswith('import '):
        # Contains code: return original
        final_answer = base_answer
        used_paraphrase = False
    else:
        # No code: paraphrase answer
        para = generate_paraphrase_with_bert2bert(base_answer)
        # semantic check
        orig_emb = embed_model.encode(base_answer, convert_to_numpy=True, normalize_embeddings=True)
        para_emb = embed_model.encode(para, convert_to_numpy=True, normalize_embeddings=True)
        sim = float(np.dot(orig_emb, para_emb))
        if len(para.split()) < len(base_answer.split()) * 0.6 or sim < 0.7:
            para = generate_with_bart(base_answer)
        final_answer = para
        used_paraphrase = True

    # Return response
    return {
        "your_question": query,
        "matched_question": matched.get("question", ""),
        "context_used": matched.get("context", matched.get("answer", "")),
        "final_answer": final_answer,
        "used_paraphrase": used_paraphrase,
        "original_answer": base_answer
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)