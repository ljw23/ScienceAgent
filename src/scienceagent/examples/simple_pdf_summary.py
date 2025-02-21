import os
import json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import dashscope
from dotenv import load_dotenv
from tqdm import tqdm
import argparse

# Load environment variables
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")


def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    full_text = "\n".join([p.page_content for p in pages])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return text_splitter.split_text(full_text)


def summarize_text(text):
    summary_prompt = "请用中文简要总结以下内容，要求分点列出核心内容：\n"
    response = dashscope.Generation.call(
        model="qwen-turbo", prompt=summary_prompt + text[:3000]
    )
    return response.output.text if response.status_code == 200 else "总结失败"


def process_pdf_and_save_summary(pdf_path, output_dir):
    output_path = os.path.join(
        output_dir, os.path.basename(pdf_path).replace(".pdf", ".json")
    )

    if Path(output_path).exists():
        return output_path

    text_chunks = load_and_split_pdf(pdf_path)
    document_summary = []

    for chunk in tqdm(text_chunks, desc=f"Processing {os.path.basename(pdf_path)}"):
        summary = summarize_text(chunk)
        document_summary.append(summary)

    combined_summary = " ".join(document_summary)
    review_prompt = "请将以下总结内容整合成一篇文献综述，包括背景、主要发现和结论：\n"
    final_summary = summarize_text(review_prompt + combined_summary[:3000])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"title": os.path.basename(pdf_path), "summary": final_summary},
            f,
            ensure_ascii=False,
            indent=4,
        )

    return output_path


def compile_summaries_to_review(summary_files):
    compiled_summaries = []

    for file_path in summary_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            compiled_summaries.append(f"{data['title']}: {data['summary']}\n")

    review_prompt = "请将以下文档总结内容整合成一篇全面的文献综述：\n"
    complete_summaries = "\n".join(compiled_summaries)
    final_review = summarize_text(review_prompt + complete_summaries[:3000])

    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(final_review)


def main(input_dir):
    pdf_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".pdf")
    ]
    output_directory = "summaries"
    os.makedirs(output_directory, exist_ok=True)

    summary_files = []
    for pdf_file in tqdm(pdf_files, desc="summary files"):
        summary_file = process_pdf_and_save_summary(pdf_file, output_directory)
        summary_files.append(summary_file)

    compile_summaries_to_review(summary_files)
    print("文献综述已保存至summary.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process PDF files and generate summaries."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="docs",
        help="The directory containing PDF files to process.",
    )
    args = parser.parse_args()

    main(args.input_dir)
