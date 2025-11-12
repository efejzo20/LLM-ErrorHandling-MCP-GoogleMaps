import os
import argparse
import asyncio
import json
from typing import List

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .rag_location_resolver import RAGLocationResolver


def _collect_files(files: List[str], directory: str | None) -> List[str]:
    collected: List[str] = []
    if files:
        for p in files:
            if os.path.exists(p):
                collected.append(p)
    if directory and os.path.isdir(directory):
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            if os.path.isfile(path) and os.path.splitext(path)[1].lower() in {".pdf", ".txt", ".md", ".markdown"}:
                collected.append(path)
    # de-duplicate, preserve order
    seen = set()
    unique = []
    for p in collected:
        if p not in seen:
            unique.append(p)
            seen.add(p)
    return unique


async def amain() -> int:
    parser = argparse.ArgumentParser(description="Demo: RAG location extraction from PDFs/text")
    parser.add_argument("--query", type=str, default="Where is the Student Society hackathon?", help="User query asking for event location")
    parser.add_argument("--files", nargs="*", default=[], help="Specific file paths to index (.pdf/.txt/.md)")
    parser.add_argument("--dir", type=str, default="PDFs", help="Directory to scan for files")
    parser.add_argument("--top-k", type=int, default=6, help="Top-k retrieved chunks")
    args = parser.parse_args()

    files = _collect_files(args.files, args.dir)
    if not files:
        print(f"No files found. Provide --files or ensure directory exists: {args.dir}")
        return 2

    resolver = RAGLocationResolver()
    chunks = await resolver.add_files(files)
    print(f"Indexed {len(files)} files into {chunks} chunks")

    result = await resolver.resolve_location(args.query, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main() -> None:
    code = asyncio.run(amain())
    raise SystemExit(code)


if __name__ == "__main__":
    main() 