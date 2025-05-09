import json

# 1. Load dữ liệu – giả sử dataset.json là một mảng JSON
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Nếu data không phải list, in ra kiểu để debug
if not isinstance(data, list):
    raise ValueError(f"Expected a list of examples, but got {type(data)}")

examples = data  # danh sách dict mỗi mục

# 2. Khởi tạo danh sách index
missing_question_idxs = []
empty_question_idxs   = []

# 3. Quét qua từng mục
for i, ex in enumerate(examples):
    # Thiếu key "question"
    if "label" not in ex:
        missing_question_idxs.append(i)
    else:
        q = ex["label"]
        # Có key nhưng rỗng hoặc chỉ whitespace
        if not isinstance(q, str) or q.strip() == "":
            empty_question_idxs.append(i)

# 4. In tóm tắt kết quả
print(f"🔍 Có {len(missing_question_idxs)} mục KHÔNG có key 'question'.")
print(f"🔍 Có {len(empty_question_idxs)} mục có 'question' nhưng rỗng hoặc chỉ whitespace.\n")

# 5. In chi tiết (tối đa 10 mục mỗi loại) để bạn debug
print("=== Mục THIẾU 'question' (tối đa 10) ===")
for idx in missing_question_idxs[:10]:
    print(f"\n--- Index {idx} ---")
    print(examples[idx])

print("\n=== Mục 'question' RỖNG (tối đa 10) ===")
for idx in empty_question_idxs[:10]:
    print(f"\n--- Index {idx} ---")
    print(examples[idx])
