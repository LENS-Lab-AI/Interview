import re
import json

def readme_to_json(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Regex to match sections like "### Q1. Question" followed by "**A:** Answer"
    pattern = re.compile(
        r"###\s*(Q\d+)\.\s*(.*?)\n\*\*A:\*\*\s*(.*?)(?=\n---|\Z)",
        re.DOTALL
    )

    qa_list = []
    for match in pattern.finditer(text):
        qnum, question, answer = match.groups()
        qa_list.append({
            "id": qnum.strip(),
            "question": question.strip(),
            "answer": answer.strip(),
            "type": "theory"
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_list, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(qa_list)} Q&A pairs to {output_path}")


readme_to_json("../README.md", "QA.json")
