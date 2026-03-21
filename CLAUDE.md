# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A full-stack RAG (Retrieval-Augmented Generation) chatbot that answers questions about course materials. Users ask questions via a web UI; the backend uses Claude with tool calling to semantically search a ChromaDB vector store and generate grounded answers.

## Project Structure

```
starting-ragchatbot-codebase/
├── backend/
│   ├── app.py                 # FastAPI server, API endpoints, startup doc loading
│   ├── rag_system.py          # Main orchestrator tying all components together
│   ├── ai_generator.py        # Anthropic API calls, tool execution loop
│   ├── vector_store.py        # ChromaDB collections, embedding, search
│   ├── document_processor.py  # File parsing, lesson splitting, chunking
│   ├── search_tools.py        # Tool ABC, CourseSearchTool, ToolManager
│   ├── session_manager.py     # In-memory conversation history per session
│   ├── models.py              # Pydantic models: Course, Lesson, CourseChunk
│   └── config.py              # All tunable parameters (single source of truth)
├── frontend/
│   ├── index.html             # Chat UI with sidebar course stats
│   ├── script.js              # Fetch API calls, session management, markdown render
│   └── style.css
├── docs/                      # Course .txt files loaded at startup
├── run.sh                     # Entry point: starts uvicorn from backend/
├── pyproject.toml
└── main.py                    # Unused
```

## Running the Application

Always use `uv` to manage all dependencies, run the server, and execute Python files. Never use `pip` or `python` directly.

```bash
# Add a dependency
uv add <package>

# Remove a dependency
uv remove <package>

# Run a Python file
uv run python <file>.py
```

```bash
# Install dependencies (first time only)
uv sync

# Start the server (from project root)
chmod +x run.sh && ./run.sh
```

App is available at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

**Required:** `.env` in the project root with no leading spaces:
```
ANTHROPIC_API_KEY=sk-ant-...
```

The backend **must** be launched via `run.sh` (which `cd`s into `backend/` first) because all Python imports are relative — e.g. `from models import Course`.

## Architecture

### Request Flow

```
User → script.js → POST /api/query → app.py → rag_system.query()
                                                    ├── session_manager (get history)
                                                    └── ai_generator.generate_response()
                                                              ├── 1st Claude call (tool_choice: auto)
                                                              │     └── stop_reason: tool_use
                                                              ├── search_tools → vector_store → ChromaDB
                                                              └── 2nd Claude call (synthesize results)
                                                    └── return (answer, sources) → frontend
```

### Key Components

**`rag_system.py`** — Wires everything together. `query()` is the single entry point: builds prompt, fetches history, calls AI, collects sources, saves exchange back to session.

**`ai_generator.py`** — Manages the two-turn Claude interaction. First call includes tool definitions; if `stop_reason == "tool_use"`, executes tools and makes a second call without tools to synthesize. Temperature is 0 (deterministic).

**`vector_store.py`** — Two ChromaDB collections: `course_catalog` (course-level metadata for fuzzy name resolution) and `course_content` (chunk-level content for semantic search). Course title is used as the document ID — duplicate titles across files are treated as the same course.

**`search_tools.py`** — `CourseSearchTool` implements the `Tool` ABC and exposes the `search_course_content` tool to Claude. Accepts optional `course_name` and `lesson_number` filters. `ToolManager` is a registry; add new tools by subclassing `Tool` and calling `tool_manager.register_tool()` in `rag_system.py`.

**`document_processor.py`** — Parses `.txt` files from `docs/`, splits by `Lesson N:` markers, chunks by sentence up to 800 chars with 100-char overlap. The first chunk of each lesson is prefixed with `Lesson N content:` for retrieval context.

**`session_manager.py`** — Pure in-memory dict. Sessions do not survive server restarts. Keeps only the last `MAX_HISTORY=2` exchanges per session.

### Document Format

Files in `docs/` must follow:
```
Course Title: <title>
Course Link: <url>
Course Instructor: <name>

Lesson 0: <title>
Lesson Link: <url>
<content...>

Lesson 1: <title>
...
```

Supported formats: `.txt`, `.pdf`, `.docx`.

### Data Persistence

- **ChromaDB** persists to `backend/chroma_db/` on disk. Already-indexed courses are skipped on restart (deduplicated by course title).
- **Conversation history** is in-memory only — lost on server restart.
- To wipe and reindex: call `rag_system.add_course_folder(path, clear_existing=True)`.

## Configuration

All parameters in `backend/config.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `CHUNK_SIZE` | 800 | Max chars per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `MAX_RESULTS` | 5 | Max chunks returned per search |
| `MAX_HISTORY` | 2 | Conversation exchanges retained per session |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Claude model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer embedding model |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB path (relative to `backend/`) |
