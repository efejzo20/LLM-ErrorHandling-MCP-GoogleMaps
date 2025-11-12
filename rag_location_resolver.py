import os
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pypdf import PdfReader


class RAGLocationResolver:
    """Lightweight RAG to extract event locations from local files (PDF/text).

    Usage:
        resolver = RAGLocationResolver()
        await resolver.add_files(["/path/to/event.pdf", "/path/to/info.txt"])  # builds retriever on first call
        result = await resolver.resolve_location("Where is the Student Council welcome event?")
        # result is a dict with keys: location, address, confidence, reasoning, sources
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None) -> None:
        self._documents: List[Document] = []
        self._retriever: Optional[BM25Retriever] = None
        # Smaller chunks with more overlap for better location extraction
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Smaller chunks = more precise location info
            chunk_overlap=200,  # More overlap = don't split location details
            add_start_index=True,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],  # Split on logical boundaries
        )
        self._llm = llm or self._create_default_llm()

    def _create_default_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            base_url=os.getenv("OPENAI_API_BASE", "https://llms-inference.innkube.fim.uni-passau.de"),
            api_key=os.getenv("UNIVERSITY_LLM_API_KEY"),
            model=os.getenv("RAG_LLM_MODEL", "llama3.1"),
            temperature=0.0,
        )

    async def add_files(self, file_paths: Sequence[str]) -> int:
        """Load and index the given files. Returns number of chunks added."""
        new_docs: List[Document] = []
        for path in file_paths:
            if not path or not os.path.exists(path):
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext == ".pdf":
                new_docs.extend(self._load_pdf(path))
            elif ext in {".txt", ".md", ".markdown"}:
                new_docs.extend(self._load_text(path))
            else:
                # Try reading as text fallback
                try:
                    new_docs.extend(self._load_text(path))
                except Exception:
                    # skip unknown format
                    continue

        if not new_docs:
            return 0

        # Split into retrievable chunks
        chunked_docs = self._text_splitter.split_documents(new_docs)
        self._documents.extend(chunked_docs)

        # Rebuild BM25 retriever with higher k for better recall
        self._retriever = BM25Retriever.from_documents(self._documents)
        self._retriever.k = 12  # Retrieve more chunks to increase chances of finding location
        return len(chunked_docs)

    def _load_pdf(self, path: str) -> List[Document]:
        reader = PdfReader(path)
        docs: List[Document] = []
        for page_index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": path,
                            "type": "pdf",
                            "page": page_index + 1,
                        },
                    )
                )
        return docs

    def _load_text(self, path: str) -> List[Document]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        if not content.strip():
            return []
        return [
            Document(
                page_content=content,
                metadata={
                    "source": path,
                    "type": "text",
                },
            )
        ]

    def _sanitize_field(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if text.lower() in {"null", "none", "n/a", "na", "unknown", "undefined", "not provided", "no location"}:
            return None
        return text

    def _expand_query(self, user_query: str) -> List[str]:
        """Generate query variations to improve retrieval."""
        queries = [user_query]
        
        # Extract potential event names (capitalized multi-word phrases)
        import re
        # Find patterns like "Student Society Hackathon", "TUM Robotics Expo"
        event_patterns = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', user_query)
        
        for event in event_patterns:
            # Add variations focused on location
            queries.append(f"{event} location")
            queries.append(f"{event} address")
            queries.append(f"{event} venue")
            queries.append(f"where is {event}")
            queries.append(f"{event} held at")
        
        # Add generic location queries
        queries.append("location address venue where")
        
        return queries[:5]  # Limit to top 5 variations
    
    async def resolve_location(self, user_query: str, *, top_k: int = 6) -> Dict[str, Any]:
        """Retrieve and extract a location from indexed files for the given query.

        Returns a dictionary:
            {
                "location": Optional[str],   # best location string for maps (place or address)
                "address": Optional[str],    # normalized address if available
                "confidence": float,         # 0..1
                "reasoning": str,
                "sources": List[Dict[str, Any]],  # [{"source": path, "page": n, "snippet": str}]
            }
        """
        if not self._retriever:
            return {
                "location": None,
                "address": None,
                "confidence": 0.0,
                "reasoning": "No documents indexed",
                "sources": [],
            }

        # Try multiple query variations to improve retrieval
        query_variations = self._expand_query(user_query)
        all_docs: List[Document] = []
        seen_content: set = set()
        
        for query_var in query_variations:
            try:
                retrieved_docs = self._retriever.invoke(query_var)  # type: ignore[arg-type]
                docs = list(retrieved_docs) if isinstance(retrieved_docs, list) else []
                # Deduplicate by content
                for doc in docs:
                    content_hash = hash(doc.page_content[:200])  # Use first 200 chars for dedup
                    if content_hash not in seen_content:
                        seen_content.add(content_hash)
                        all_docs.append(doc)
            except Exception:
                continue
        
        # Take top_k unique documents
        retrieved: List[Document] = all_docs[:top_k]
        if not retrieved:
            return {
                "location": None,
                "address": None,
                "confidence": 0.0,
                "reasoning": "No relevant evidence found",
                "sources": [],
            }

        sources_preview: List[Dict[str, Any]] = []
        for d in retrieved:
            snippet = d.page_content
            snippet = snippet.strip().replace("\n", " ")
            if len(snippet) > 500:
                snippet = snippet[:500] + "…"
            src = {
                "source": d.metadata.get("source"),
                "page": d.metadata.get("page"),
                "snippet": snippet,
            }
            sources_preview.append(src)

        system_prompt = (
            "You are a precise location extraction assistant. Extract the PHYSICAL LOCATION where an event is held.\n\n"
            "CRITICAL RULES:\n"
            "1. Look for ADDRESSES: Street names, building names, room numbers, postal codes, city names\n"
            "2. Extract VENUE DETAILS: 'Innovation Hub Building C Room 101', 'BMW Welt Am Olympiapark 1'\n"
            "3. DO NOT just repeat the event name - find WHERE it is located\n"
            "4. If evidence mentions 'Location:', 'Venue:', 'Where:', 'Address:' - extract what follows\n"
            "5. Combine building + street + city for best results\n\n"
            "Return ONLY valid JSON (no code fences):\n"
            "{\n"
            "  \"location\": \"Full venue/building with street and city\" | null,\n"
            "  \"address\": \"Normalized postal address\" | null,\n"
            "  \"confidence\": 0.0-1.0 (use 0.0-0.3 if only event name found, 0.8+ if full address),\n"
            "  \"reasoning\": \"Brief explanation (<30 words)\"\n"
            "}\n\n"
            "EXAMPLES:\n"
            "Good: {\"location\": \"Innovation Hub, Building C, Room 101, University of Berlin\", \"confidence\": 0.9}\n"
            "Bad: {\"location\": \"Student Hackathon\", \"confidence\": 0.2} // This is just the event name!\n"
            "Good: {\"location\": \"BMW Welt, Am Olympiapark 1, 80809 Munich\", \"confidence\": 0.95}\n\n"
            "If NO physical location found in evidence: {\"location\": null, \"address\": null, \"confidence\": 0.0}"
        )

        evidence_text = "\n\n".join(
            [
                f"Source: {s.get('source')} Page: {s.get('page')}\nSnippet: {s.get('snippet')}"
                for s in sources_preview
            ]
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"User query: {user_query}\n\n"
                    f"Evidence:\n{evidence_text}\n\n"
                    "Respond with JSON only."
                )
            ),
        ]

        response = await self._llm.ainvoke(messages)
        content = self._clean_json_response(response.content)

        result: Dict[str, Any]
        try:
            result = json.loads(content)
        except Exception:
            result = {
                "location": None,
                "address": None,
                "confidence": 0.0,
                "reasoning": "Failed to parse LLM output",
            }

        # Normalize result shape and attach sources
        normalized_location = self._sanitize_field(result.get("location"))
        normalized_address = self._sanitize_field(result.get("address"))
        confidence_value = result.get("confidence", 0.0)
        try:
            confidence_float = float(confidence_value) if confidence_value is not None else 0.0
        except Exception:
            confidence_float = 0.0

        # VALIDATION: Check if location looks like just the event name (not an address)
        if normalized_location:
            location_lower = normalized_location.lower()
            # Red flags: location is just event keywords without address components
            event_only_keywords = ["hackathon", "conference", "fair", "expo", "festival", "ceremony", "summit", "show"]
            has_address_markers = any(marker in location_lower for marker in [
                "street", "str.", "straße", "avenue", "ave", "road", "rd", "building", 
                "floor", "room", "suite", "#", "platz", "weg", "gasse"
            ])
            is_just_event = any(kw in location_lower for kw in event_only_keywords) and not has_address_markers
            
            # Also check if location is suspiciously short (< 3 words = likely just event name)
            word_count = len(normalized_location.split())
            
            if is_just_event and word_count < 5:
                # Likely just the event name, not a real location
                print(f"⚠️  RAG validation: '{normalized_location}' looks like event name, not address")
                confidence_float = min(confidence_float, 0.3)  # Cap confidence at 0.3

        normalized = {
            "location": normalized_location,
            "address": normalized_address,
            "confidence": confidence_float,
            "reasoning": result.get("reasoning") or "",
            "sources": sources_preview,
        }
        return normalized

    def _clean_json_response(self, raw: str) -> str:
        content = raw.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()


__all__ = ["RAGLocationResolver"] 