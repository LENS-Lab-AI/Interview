import re
import json
import os

def extract_qa_from_markdown(file_path):
    """Extract Q&A pairs from a single markdown file."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Skip empty files
    if not text:
        print(f"Skipping empty file: {file_path}")
        return []

    # Extract type from first line (e.g. '# Theory')
    first_line = text.splitlines()[0].strip()
    type_match = re.match(r"#\s*(.*)", first_line)
    topic_type = type_match.group(1).strip() if type_match else "Unknown"

    # Match Qn and A sections
    pattern = re.compile(
        r"###\s*(Q\d+)\.\s*(.*?)\n\*\*A:\*\*\s*(.*?)(?=\n---|\Z)",
        re.DOTALL
    )

    qa_list = []
    for match in pattern.finditer(text):
        qnum, question, answer = match.groups()
        qa_list.append({
            "type": topic_type,
            "id": qnum.strip(),
            "question": question.strip(),
            "answer": answer.strip()
        })

    if qa_list:
        print(f"Extracted {len(qa_list)} Q&A from {os.path.basename(file_path)}")
    else:
        print(f"No Q&A found in {os.path.basename(file_path)}")

    return qa_list


def convert_all_markdowns(input_dir, output_file):
    """Read all markdowns from a folder and merge QAs into one JSON."""
    all_qas = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".md"):
            file_path = os.path.join(input_dir, filename)
            qas = extract_qa_from_markdown(file_path)
            all_qas.extend(qas)

    if not all_qas:
        print("No Q&A found in any markdown files.")
        return

    with open(f"web_utils/{output_file}", "w", encoding="utf-8") as f:
        json.dump(all_qas, f, ensure_ascii=False, indent=2)

    print(f"\nAll done! Exported {len(all_qas)} Q&A pairs to {output_file}")


if __name__ == "__main__":
    input_dir = "Topics/Transformers"
    output_file = "QA.json"
    convert_all_markdowns(input_dir, output_file)
