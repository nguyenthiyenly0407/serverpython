from transformers import pipeline
import json

# Tạo pipeline tóm tắt
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Đọc file JSON chứa dataset
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Thêm trường key_answer vào từng mẫu
for item in data:
    context = item.get("answer", "")
    if context.strip():  # chỉ tóm tắt nếu có context
        summary = summarizer(context, max_length=50, min_length=15, do_sample=False)
        item["key_answer"] = summary[0]["summary_text"]
    else:
        item["key_answer"] = ""

# Lưu lại file mới
with open("dataset_answer.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)