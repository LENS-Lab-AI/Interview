import re
import json
import os
import argparse

def extract_qa_from_markdown(file_path, verbose=False):
    """Extract Q&A pairs from a single markdown file."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        if verbose:
            print(f"Skipping empty file: {file_path}")
        return []

    # Extract type from first line (e.g. '# Theory')
    first_line = text.splitlines()[0].strip()
    type_match = re.match(r"#\s*(.*)", first_line)
    topic_type = type_match.group(1).strip() if type_match else "Unknown"

    # Match Qn and A sections (handles Q1. or Q1:)
    pattern = re.compile(
        r"###\s*(Q\d+)[\.:]?\s*(.*?)\n\*\*A:\*\*\s*(.*?)(?=\n###|\n##|\Z)",
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

    if verbose:
        if qa_list:
            print(f"Extracted {len(qa_list)} Q&A from {os.path.basename(file_path)} ({topic_type})")
        else:
            print(f"No Q&A found in {os.path.basename(file_path)}")

    return qa_list


def convert_all_markdowns(input_dir, verbose=False):
    """Read all markdowns from a folder and merge QAs into one JSON."""
    all_qas = []

    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return

    for filename in os.listdir(input_dir):
        if filename.endswith(".md"):
            file_path = os.path.join(input_dir, filename)
            qas = extract_qa_from_markdown(file_path, verbose)
            all_qas.extend(qas)

    if not all_qas:
        print("No Q&A found in any markdown files.")
        return
    output_file = os.path.join(input_dir, "QA.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_qas, f, ensure_ascii=False, indent=2)

    print(f"\nExported {len(all_qas)} Q&A pairs to {output_file}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert multiple Markdown Q&A files into a single JSON file."
    )
    parser.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Directory containing markdown (.md) files."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable detailed logging."
    )

    args = parser.parse_args()
    convert_all_markdowns(args.input_dir, args.verbose)


if __name__ == "__main__":
    main()
