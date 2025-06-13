import json

# Đọc dữ liệu từ file JSON gốc
with open("long_answers.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Thêm key_answer cho từng entry
for item in data:
    item["key_answer"] = item["answer"]  # Hoặc xử lý tạo paraphrase tại đây nếu cần

# Ghi lại file JSON mới
with open("longkey.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
