# Google Maps MCP Evaluator with Intelligent Error Recovery

A comprehensive evaluation framework for testing LLM-based function calling with the [Google Maps Model Context Protocol (MCP) server](https://github.com/modelcontextprotocol/servers-archived/tree/main/src/google-maps). Features intelligent error recovery using both RAG (Retrieval-Augmented Generation) for event location resolution and LLM-based parameter correction.

## 🎯 Overview

This project evaluates how well different LLMs can:
- Select the correct Google Maps API tool for a given query
- Generate appropriate parameters for API calls
- Recover from errors using RAG or error-aware retry mechanisms
- Handle complex queries involving events, locations, and directions

### Key Features

- **Hybrid Error Recovery**: Intelligently routes between RAG (for location/event errors) and LLM-based error recovery (for parameter/API errors)
- **RAG Event Resolution**: Extracts event locations from local PDFs/documents using BM25 retrieval + LLM extraction
- **LLM-as-a-Judge**: Automated evaluation of tool selection, parameters, and result quality
- **Multi-Model Support**: Test with OpenAI, Ollama, or university-hosted LLMs
- **Comprehensive Logging**: Detailed JSONL output for every query with success/failure tracking

---

## 📁 Project Structure

```
.
├── src/                             # Main source code
│   ├── evaluators/                  # Batch evaluation scripts
│   │   ├── maps_evaluator.py        # Basic evaluator (no recovery)
│   │   ├── maps_evaluator_error.py  # Error recovery with LLM
│   │   ├── maps_evaluator_rag.py    # RAG fallback for events
│   │   └── maps_evaluator_hybrid.py # 🌟 Hybrid (RAG + Error recovery)
│   │
│   ├── judges/                      # Result evaluation
│   │   ├── maps_llm_judge.py        # General tool-calling judge
│   │   └── maps_hybrid_judge.py     # RAG-aware judge with ground truth
│   │
│   ├── agents/                      # Interactive agents
│   │   ├── maps_direct_flow.py      # Interactive single-query agent
│   │   └──  maps_direct_flow_rag.py  # Interactive agent with RAG
│   │
│   └── utils/                       # Helper modules
│       ├── error_recovery.py        # LLM-based error correction
│       ├── rag_location_resolver.py # RAG location extraction
│       ├── rag_demo.py              # RAG testing demo
│       └── maps_dataset_generator.py # Generate test queries
│
├── Dataset/                         # Organized test queries
│   ├── MCP-GoogleMaps/              # Google Maps evaluation queries
│   │   ├── maps_queries.txt         # Main query set
│   │   └── Test-Data/               # Additional test sets
│   └── RAG-Location/                # RAG-specific queries and ground truth
│       ├── rag_queries.txt          # Event-based queries
│       ├── rag_expected_locations.csv # Ground truth for RAG evaluation
│       └── Test-Data/               # Extended RAG test queries
│
├── PDFs/                            # Mock event documents for RAG testing
│   ├── TUM_Robotics_Expo.pdf
│   ├── Berlin_Startup_Fair.pdf
│   └── ... (more event PDFs)
│
├── DATA-EVAL/                       # Raw evaluation results (JSONL)
├── Data-Sumary/                     # Processed summaries and statistics
├── results_rag/                     # RAG-specific results
│
├── README.md
├── requirements.txt
└── uv.lock
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js (for Google Maps MCP server)
- Google Maps API key
- OpenAI API key or Ollama installation

### Installation

1. **Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file in the root directory:
```bash
# Required
GOOGLE_MAPS_API_KEY=your_google_maps_api_key_here

# Choose one LLM provider:
# Option 1: OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Option 2: University LLM endpoint
UNIVERSITY_LLM_API_KEY=your_university_api_key_here
OPENAI_API_BASE=https://llms-inference.innkube.fim.uni-passau.de

# Option 3: Ollama (local)
# Just ensure Ollama is running: ollama serve
```

4. **Install Google Maps MCP server**

The project uses the [Google Maps MCP server](https://github.com/modelcontextprotocol/servers-archived/tree/main/src/google-maps) which provides access to Google Maps APIs through the Model Context Protocol.

```bash
npx -y @modelcontextprotocol/server-google-maps
```

## 📖 Usage Guide

### 1. Generate Test Queries

Create a diverse set of test queries for evaluation:

```bash
python -m src.utils.maps_dataset_generator \
  --num 200 \
  --output Dataset/MCP-GoogleMaps/my_queries.txt \
  --format txt
```

---

### 2. Run Evaluations

#### **Option A: Hybrid Evaluator**

The most sophisticated evaluator with intelligent routing between RAG and error recovery:

```bash
python -m src.evaluators.maps_evaluator_hybrid \
  --queries-file Dataset/MCP-GoogleMaps/maps_queries.txt \
  --output results_hybrid.jsonl \
  --rag-dir PDFs
```

**What it does:**
- Initial attempt: LLM generates tool call → executes on MCP
- On failure: Routes to RAG (if location issue) or error recovery (if parameter issue)
- Cross-retry: If first recovery fails with different error type, tries the other route
- Logs everything: Initial attempt, recovery attempts, final success/failure

**Options:**
- `--queries-file`: Path to queries (one per line)
- `--query`: Single query for testing (repeatable)
- `--queries-csv`: CSV file with query column
- `--output`: Output JSONL path (default: `maps_eval_results_hybrid.jsonl`)
- `--rag-dir`: Directory containing event PDFs (default: `PDFs`)
- `--rag-file`: Specific PDF file (repeatable)
- `--stream`: Write results as they complete

#### **Option B: RAG-Only Evaluator**

For queries specifically involving event locations:

```bash
python -m src.evaluators.maps_evaluator_rag \
  --queries-file Dataset/RAG-Location/rag_queries.txt \
  --output results_rag.jsonl \
  --rag-dir PDFs
```

#### **Option C: Error Recovery Evaluator**

For general parameter/API error testing:

```bash
python -m src.evaluators.maps_evaluator_error \
  --queries-file Dataset/MCP-GoogleMaps/maps_queries.txt \
  --output results_error.jsonl
```

#### **Option D: Basic Evaluator**

Baseline without any recovery (to measure improvement):

```bash
python -m src.evaluators.maps_evaluator \
  --queries-file Dataset/MCP-GoogleMaps/maps_queries.txt \
  --output results_basic.jsonl
```

---

### 3. Evaluate Results with LLM Judge

#### **Hybrid Judge** (for RAG evaluation with ground truth)

```bash
python -m src.judges.maps_hybrid_judge \
  --input results_hybrid.jsonl \
  --expected-locations Dataset/RAG-Location/rag_expected_locations.csv \
  --output judged_hybrid.jsonl \
  --summary hybrid_summary.json \
  --model gpt-4o-mini
```

**Evaluates:**
- ✅ RAG Trigger: Was RAG triggered when needed?
- ✅ RAG Resolution: Did RAG find the correct location?
- ✅ RAG Usage: Was resolved location used in retry?
- ✅ Final Outcome: Did the query succeed?

**Options:**
- `--model gpt-4o-mini`: Use OpenAI model
- `--model ollama:phi4:14b`: Use local Ollama model
- `--limit 10`: Judge only first 10 records
- `--max-concurrency 8`: Parallel LLM calls

#### **General Judge** (for tool selection evaluation)

```bash
python -m src.judges.maps_llm_judge \
  --input results_hybrid.jsonl \
  --output judged_general.jsonl \
  --summary general_summary.json
```

**Evaluates:**
- Tool selection correctness
- Parameter appropriateness
- Result relevance to query

---

### 4. Interactive Testing

Test individual queries interactively with the direct flow agents:

#### **Basic Direct Flow**
```bash
python -m src.agents.maps_direct_flow
# Then enter queries interactively
```

#### **RAG-Enhanced Direct Flow**
```bash
python -m src.agents.maps_direct_flow_rag --rag-dir PDFs
# Test event queries with RAG fallback
```

## 🔧 Advanced Usage

### Testing RAG Location Resolution

Test RAG independently:

```bash
python -m src.utils.rag_demo \
  --query "Where is the TUM Robotics Expo?" \
  --dir PDFs \
  --top-k 6
```

### Custom Event PDFs

Add your own event PDFs to the `PDFs/` directory. Include:
- Event name in the document
- Complete address or location details
- Date/time information

### Modifying LLM Models

Edit the evaluator files to change models:

**For OpenAI:**
```python
self.llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o",  # or "gpt-4o-mini"
    temperature=0.0,
)
```

**For Ollama:**
```python
from langchain_community.chat_models import ChatOllama

self.llm = ChatOllama(
    base_url="http://localhost:11434",
    model="phi4:14b",  # or "llama3.1", "qwen2"
    temperature=0.0,
)
```

---

## 📊 Understanding Output Files

### Evaluation Results (JSONL)

Each line contains:
```json
{
  "query": "Get directions to TUM Robotics Expo from Munich HBF",
  "tool_call": {
    "tool_name": "maps_directions",
    "parameters": {"origin": "Munich HBF", "destination": null},
    "reasoning": "Event detected, setting destination to null"
  },
  "tool_response": "Error: destination required",
  "success": false,
  "error": "Missing destination",
  "rag_used": true,
  "rag_resolved_location": "TUM Main Campus, Mechanical Engineering Building",
  "retry_tool_call": {
    "tool_name": "maps_directions",
    "parameters": {
      "origin": "Munich HBF",
      "destination": "TUM Main Campus, Mechanical Engineering Building"
    }
  },
  "retry_success": true,
  "final_success": true
}
```

### Summary Statistics (JSON)

```json
{
  "count": 100,
  "initial_success": 65,
  "rag_attempts": 30,
  "rag_recovered": 25,
  "error_recovery_attempts": 10,
  "error_recovered": 8,
  "final_success": 98,
  "pass_rate": 0.98,
  "avg_scores": {
    "rag_trigger": 0.95,
    "rag_resolution": 0.88,
    "rag_usage": 0.90,
    "overall": 0.91
  }
}
```

---

## 🎓 Key Concepts

### **Hybrid Error Recovery**

The hybrid evaluator intelligently routes errors:

1. **NOT_FOUND / Location Errors** → RAG Resolution
   - Example: "TUM Robotics Expo" not recognized
   - Action: Search PDFs for event location
   - Retry: Regenerate tool call with resolved address

2. **Parameter / API Errors** → Error Recovery
   - Example: Wrong parameter format, missing fields
   - Action: LLM analyzes error and corrects parameters
   - Retry: Execute with corrected parameters

3. **Cross-Retry Logic**
   - If RAG retry fails with parameter error → Try error recovery
   - If error recovery fails with NOT_FOUND → Try RAG
   - Maximizes recovery success rate

### **RAG Location Resolution**

1. Index event PDFs using BM25 retrieval
2. Retrieve top-k relevant chunks for query
3. LLM extracts location information from chunks
4. Returns: address, confidence score, reasoning, sources

---

## 📈 Typical Workflow

```bash
# 1. Generate diverse test queries
python -m src.utils.maps_dataset_generator --num 200 --output Dataset/MCP-GoogleMaps/queries_200.txt

# 2. Run hybrid evaluation
python -m src.evaluators.maps_evaluator_hybrid \
  --queries-file Dataset/MCP-GoogleMaps/queries_200.txt \
  --output results.jsonl \
  --rag-dir PDFs

# 3. Judge results with ground truth
python -m src.judges.maps_hybrid_judge \
  --input results.jsonl \
  --expected-locations Dataset/RAG-Location/rag_expected_locations.csv \
  --output judged.jsonl \
  --summary summary.json

# 4. View summary
cat summary.json | jq

# 5. Analyze in notebook
jupyter notebook table.ipynb
```


## 🐛 Troubleshooting

### "Google Maps API Key not found"
```bash
# Check your .env file
cat .env | grep GOOGLE_MAPS_API_KEY

# Or export directly
export GOOGLE_MAPS_API_KEY="your_key_here"
```

### "MCP server connection failed"
```bash
# Test MCP server manually
npx -y @modelcontextprotocol/server-google-maps

# Check if it responds (Ctrl+C to exit)
```

**Additional help:** See the [Google Maps MCP server documentation](https://github.com/modelcontextprotocol/servers-archived/tree/main/src/google-maps) for troubleshooting and configuration details.

### "No RAG files found"
```bash
# Ensure PDFs directory exists and contains files
ls -la PDFs/

# Verify PDF format
file PDFs/TUM_Robotics_Expo.pdf
```

### "Ollama model not found"
```bash
# List available models
ollama list

# Pull model if needed
ollama pull phi4:14b
```


