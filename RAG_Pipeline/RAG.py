  """
  Combined RAG pipeline for local PDFs, arXiv results, and GitHub repositories.

  The LangGraph agent can discover arXiv papers and GitHub repositories. This
  module turns those retrieval outputs into searchable chunks in the same Qdrant
  collection used by the PDF RAG pipeline.
  """

  from __future__ import annotations
  import arxiv
  from pydantic import BaseModel, Field
  import os
  import json
  import re
  from operator import add
  from rank_bm25 import BM25Okapi
  import time
  import urllib
  import urllib.request
  import urllib.parse
  import urllib.error
  import sys
  from typing_extensions import TypedDict
  import uuid
  from langgraph.graph import StateGraph, START, END
  from typing import Any, Dict, List, Optional, Sequence,Annotated
  from groq import Groq
  from sentence_transformers import CrossEncoder
  from langchain_text_splitters import (
      MarkdownHeaderTextSplitter,
      RecursiveCharacterTextSplitter,
  )
  import random
  from langchain_groq import ChatGroq
  from langchain_core.prompts import ChatPromptTemplate
  from llama_parse import LlamaParse
  from qdrant_client import QdrantClient
  from qdrant_client.models import (
      Distance,
      FieldCondition,
      Filter,
      MatchAny,
      MatchValue,
      PointStruct,
      VectorParams,
  )
  from sentence_transformers import SentenceTransformer
  PDF_STORAGE_DIR = "downloaded_papers"
  os.makedirs(PDF_STORAGE_DIR, exist_ok=True)
  from langchain_google_genai import ChatGoogleGenerativeAI

  from langchain_cerebras import ChatCerebras


  def _get_secret(name: str):
      from google.colab import userdata
      return userdata.get(name)

  def _safe_filename(arxiv_url: str) -> str:
      arxiv_id = arxiv_url.rstrip("/").split("/")[-1]
      return re.sub(r"[^\w\.-]", "_", arxiv_id) + ".pdf"



  llm = ChatGroq(
      model="llama-3.3-70b-versatile",
      api_key=_get_secret("GROQ_API_KEY"),
      temperature=0.2
  )
  # Configurations and Keys setup
  LLAMA_API_KEY = _get_secret("LLAMA_API_KEY")
  GEMINI_API_KEY = _get_secret("GEMINI_API_KEY")
  GROQ_API_KEY = _get_secret("GROQ_API_KEY")
  GITHUB_PERSONAL_ACCESS_TOKEN = _get_secret("GITHUB_PERSONAL_ACCESS_TOKEN")

  COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "research_memory")
  QDRANT_PATH = os.getenv("QDRANT_PATH", "qdrant_db")

  embed_model = SentenceTransformer("allenai-specter")
  VECTOR_DIM = embed_model.get_embedding_dimension()
  qdrant = QdrantClient(":memory:")

  cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

  class ResearchState(TypedDict):
    query: str
    pdf_path: str | None
    paper_id: str | None
    paper_title: str | None
    markdown: str | None
    chunks: list[Any]
    problem_summary: str
    github_queries: list[str]
    arxiv_queries: list[str]
    papers: Annotated[list[dict],add]
    downloaded_papers: list[dict]       
    paper_chunks: Annotated[list[Any],add]             
    indexed_research: dict              
    repositories: Annotated[list[dict],add]
    relevance_score: int
    feedback: str | None
    retrieved_context: list[dict]
    answer: str | None
    loop_count: int

  class OrchestratorResponse(BaseModel):
      problem_summary: str = Field(description="A concise summary of the core problem.")
      github_queries: list[str] = Field(description="3-5 optimized keywords/queries for GitHub repository search.")
      arxiv_queries: list[str] = Field(description="3-5 optimized keywords/queries for arXiv paper search.")

  class CriticResponse(BaseModel):
      relevance_score: int = Field(description="An integer between 0 and 100.")
      feedback : str = Field(description="Brief, constructive feedback focusing on actionable search terms or direction for the next iteration.")


  def ensure_collection(collection_name: str = COLLECTION_NAME) -> None:
      """Create the Qdrant collection when it does not already exist."""
      if qdrant.collection_exists(collection_name):
          return

      qdrant.create_collection(
          collection_name=collection_name,
          vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
      )


  ensure_collection()


  def _point_id(source_type: str, source_id: str, chunk_index: int) -> str:
      key = f"{source_type}:{source_id}:{chunk_index}"
      return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


  def _split_text(text: str, metadata: Dict[str, Any]) -> List[Any]:
      splitter = RecursiveCharacterTextSplitter(
          chunk_size=800,
          chunk_overlap=120,
          separators=["\n\n", "\n", ". ", " "],
      )
      chunks = splitter.create_documents([text], metadatas=[metadata])
      for index, chunk in enumerate(chunks):
          chunk.metadata["chunk_index"] = index
      return chunks


  def _upsert_chunks(
      chunks: Sequence[Any],
      collection_name: str = COLLECTION_NAME,
  ) -> int:
      if not chunks:
          return 0

      texts = [chunk.page_content for chunk in chunks]
      embeddings = embed_model.encode(
          texts,
          show_progress_bar=True,
          batch_size=32,
      ).tolist()

      points = []
      for chunk, text, vector in zip(chunks, texts, embeddings):
          metadata = dict(chunk.metadata)
          source_type = metadata.get("source_type", "document")
          source_id = metadata.get("source_id") or metadata.get("url") or text[:80]
          chunk_index = metadata.get("chunk_index", 0)
          points.append(
              PointStruct(
                  id=_point_id(source_type, str(source_id), int(chunk_index)),
                  vector=vector,
                  payload={"text": text, **metadata},
              )
          )

      qdrant.upsert(collection_name=collection_name, points=points)
      return len(points)


  # Step 1: Convert PDF to Markdown via LlamaParse.
  def convert_pdf(pdf_path: str) -> str:
      """Takes a path to a PDF and returns parsed Markdown."""

      parser = LlamaParse(
          api_key=LLAMA_API_KEY,
          result_type="markdown",
          verbose=True,
      )
      documents = parser.load_data(pdf_path)
      return "\n\n".join(doc.text for doc in documents)


  # Use to convert orginal problem statement into markdown
  def parse_pdf_node(state: ResearchState) -> dict:
      markdown = convert_pdf(state["pdf_path"])
      return {"markdown": markdown}

  # Step 2: Section-aware chunking for LlamaParse Markdown output.
  def chunk_markdown(markdown_text: str, paper_id: str, paper_title: str = "") -> List[Any]:
      """Split Markdown by paper sections, then into embedding-sized chunks."""
      header_splitter = MarkdownHeaderTextSplitter(
          headers_to_split_on=[
              ("#", "title"),
              ("##", "section"),
              ("###", "subsection"),
          ],
          strip_headers=False,
      )
      header_chunks = header_splitter.split_text(markdown_text)

      char_splitter = RecursiveCharacterTextSplitter(
          chunk_size=800,
          chunk_overlap=120,
          separators=["\n\n", "\n", ". ", " "],
      )
      final_chunks = char_splitter.split_documents(header_chunks)

      for index, chunk in enumerate(final_chunks):
          chunk.metadata.update(
              {
                  "source_type": "pdf",
                  "source_id": paper_id,
                  "paper_id": paper_id,
                  "paper_title": paper_title,
                  "title": paper_title,
                  "chunk_index": index,
                  "chunk_id": f"{paper_id}_{index}"
              }
          )
      return final_chunks


  def download_pdf(pdf_url: str, arxiv_url: str, storage_dir: str = PDF_STORAGE_DIR, timeout: int = 30) -> str | None:
      filename = _safe_filename(arxiv_url)
      local_path = os.path.join(storage_dir, filename)

      if os.path.exists(local_path):
          return local_path

      request = urllib.request.Request(
          pdf_url,
          headers={"User-Agent": "Mozilla/5.0 (research-agent)"},
      )
      try:
          with urllib.request.urlopen(request, timeout=timeout) as response:
              data = response.read()
          with open(local_path, "wb") as f:
              f.write(data)
          return local_path
      except Exception as exc:
          print(f"Failed to download {pdf_url}: {exc}")
          return None

  def download_all_papers(papers: list[dict], delay_seconds: float = 3.0) -> list[dict]:
      """Downloads PDFs with a delay between requests to respect arXiv's rate limits."""
      downloaded = []
      for i, paper in enumerate(papers):
          path = download_pdf(paper["pdf_url"], paper["arxiv_url"])
          if path:
              downloaded.append({**paper, "local_path": path})
          if i < len(papers) - 1:
              time.sleep(delay_seconds)
      return downloaded



  # Orchestrator Node, classifes intent and generates specialized ps for github & arxiv retreival
  def orchestrator_node(state: ResearchState) -> dict:
      prompt_template = ChatPromptTemplate.from_messages([
          ("system", ORCHESTRATOR_PROMPT),
          ("user", "{query}")
      ])

      messages = prompt_template.format_messages(
          problem_statement=problem_statement,
          query=state["query"],
          feedback=state.get("feedback", "")
      )

      response = llm.invoke(messages)   # plain invoke, no with_structured_output, because of inconsistency with groq models
      cleaned = response.content.strip()

      try:
          parsed = json.loads(cleaned)
      except json.JSONDecodeError:
          match = re.search(r'\{.*\}', cleaned, re.DOTALL)
          if not match:
              raise ValueError(f"Could not parse orchestrator output: {cleaned[:300]}")
          parsed = json.loads(match.group())

      return OrchestratorResponse(**parsed).model_dump()
  import requests  # arxiv depends on it; used for network-error classification

  MAX_ARXIV_QUERIES = 1
  MAX_ARXIV_RESULTS_PER_QUERY = 5
  ARXIV_DELAY_SECONDS = 12.0          # polite spacing between successful requests
  ARXIV_MAX_RETRIES = 5
  ARXIV_RETRY_BASE_SECONDS = 12.0     # backoff base (decoupled from inter-query delay)
  ARXIV_RETRY_CAP_SECONDS = 300.0     # ceiling on any single backoff sleep


  def _is_retryable_arxiv_error(exc: Exception) -> bool:
      # Empty feed = arXiv's usual "overloaded / throttling you" signal (HTTP 200, no entries).
      if isinstance(exc, arxiv.UnexpectedEmptyPageError):
          return True
      # Retry 429 + 5xx; do NOT retry 4xx like 400 (bad query) — retrying just wastes requests.
      if isinstance(exc, arxiv.HTTPError):
          status = getattr(exc, "status", None)
          return status is None or status == 429 or (500 <= status <= 599)
      # Transient network layer: timeouts, connection resets, DNS blips.
      if isinstance(exc, requests.exceptions.RequestException):
          return True
      return False


  def _backoff_seconds(attempt: int) -> float:
      # Exponential with full jitter, floored at half the window so we never retry near-instantly.
      window = min(ARXIV_RETRY_CAP_SECONDS, ARXIV_RETRY_BASE_SECONDS * (2 ** attempt))
      return window / 2 + random.uniform(0, window / 2)


  def arxiv_search_node(state: ResearchState) -> dict:
      client = arxiv.Client(
          page_size=MAX_ARXIV_RESULTS_PER_QUERY,
          delay_seconds=ARXIV_DELAY_SECONDS,
          num_retries=0,  # manual retry handling below
      )

      papers = []
      seen_urls = set()

      # No fallback queries. Only use orchestrator-generated queries.
      queries = state.get("arxiv_queries", [])[:MAX_ARXIV_QUERIES]

      if not queries:
          print("No arXiv queries provided by orchestrator.")
          return {"papers": papers}

      for query_index, query in enumerate(queries):
          if query_index > 0:
              print(f"Sleeping {ARXIV_DELAY_SECONDS}s before next arXiv query.")
              time.sleep(ARXIV_DELAY_SECONDS)

          search = arxiv.Search(
              query=query,
              max_results=MAX_ARXIV_RESULTS_PER_QUERY,
              sort_by=arxiv.SortCriterion.Relevance,
          )

          query_success = False

          for attempt in range(ARXIV_MAX_RETRIES):
              try:
                  print(f"arXiv attempt {attempt + 1}/{ARXIV_MAX_RETRIES}: {query}")

                  results = list(client.results(search))

                  for result in results:
                      if result.entry_id in seen_urls:
                          continue

                      pdf_url = getattr(result, "pdf_url", None)
                      if not pdf_url:
                          print(f"Skipping paper without PDF URL: {result.title}")
                          continue

                      seen_urls.add(result.entry_id)
                      papers.append(
                          {
                              "title": result.title,
                              "authors": [author.name for author in result.authors],
                              "summary": result.summary.replace("\n", " "),
                              "categories": result.categories,
                              "arxiv_url": result.entry_id,
                              "pdf_url": pdf_url,
                              "source_query": query,
                          }
                      )

                  query_success = True
                  break

              except Exception as exc:
                  if not _is_retryable_arxiv_error(exc):
                      print(f"Non-retryable arXiv error for query '{query}': {type(exc).__name__}: {exc}")
                      break

                  # Retryable, but we're out of attempts.
                  if attempt == ARXIV_MAX_RETRIES - 1:
                      print(f"Giving up after {ARXIV_MAX_RETRIES} attempts for query '{query}': {exc}")
                      break

                  sleep_for = _backoff_seconds(attempt)
                  print(
                      f"Retryable arXiv error ({type(exc).__name__}) for query '{query}'. "
                      f"Backing off {sleep_for:.1f}s (attempt {attempt + 1}). Error: {exc}"
                  )
                  time.sleep(sleep_for)

          if not query_success:
              print(f"Query failed after retries: {query}")

      print(f"Collected {len(papers)} arXiv papers total.")
      return {"papers": papers}

  GITHUB_REQUEST_DELAY_SECONDS = 2.0

  def github_search_node(state: ResearchState) -> dict:
      """Search GitHub repositories using agent-generated GitHub queries."""
      repositories = []
      seen_urls = set()
      queries = state.get("github_queries", [])
      token = GITHUB_PERSONAL_ACCESS_TOKEN

      if not token:
          print("WARNING: GITHUB_PERSONAL_ACCESS_TOKEN is missing/empty — "
                "falling back to unauthenticated GitHub search (10 req/min limit).")

      for i, query in enumerate(queries):
          if i > 0:
              time.sleep(GITHUB_REQUEST_DELAY_SECONDS)

          params = urllib.parse.urlencode({"q": query, "per_page": 3})
          request = urllib.request.Request(
              f"https://api.github.com/search/repositories?{params}",
              headers={
                  "Accept": "application/vnd.github+json",
                  "User-Agent": "research-agent",              # GitHub 403s on requests without this
                  "X-GitHub-Api-Version": "2022-11-28",
                  **({"Authorization": f"Bearer {token}"} if token else {}),
              },
          )
          try:
              with urllib.request.urlopen(request, timeout=30) as response:
                  payload = response.read().decode("utf-8")
          except urllib.error.HTTPError as exc:
              body = exc.read().decode("utf-8", errors="ignore")
              print(f"GitHub search failed for '{query}': {exc} — {body[:300]}")
              # Secondary/abuse rate limit sends a Retry-After header — respect it once.
              retry_after = exc.headers.get("Retry-After") if exc.headers else None
              if exc.code == 403 and retry_after:
                  wait = float(retry_after)
                  print(f"Rate-limited by GitHub, sleeping {wait:.0f}s and retrying once.")
                  time.sleep(wait)
                  try:
                      with urllib.request.urlopen(request, timeout=30) as response:
                          payload = response.read().decode("utf-8")
                  except urllib.error.HTTPError as exc2:
                      body2 = exc2.read().decode("utf-8", errors="ignore")
                      print(f"Retry also failed for '{query}': {exc2} — {body2[:300]}")
                      continue
              else:
                  continue
          except Exception as exc:
              print(f"GitHub search unexpected error for '{query}': {exc}")
              continue

          for item in json.loads(payload).get("items", []):
              url = item.get("html_url", "")
              if url in seen_urls:
                  continue

              seen_urls.add(url)
              repositories.append(
                  {
                      "name": item.get("full_name", ""),
                      "description": item.get("description", "") or "",
                      "language": item.get("language", "") or "",
                      "topics": item.get("topics", []),
                      "stars": item.get("stargazers_count", 0),
                      "forks": item.get("forks_count", 0),
                      "url": url,
                      "source_query": query,
                  }
              )

      print(f"Collected {len(repositories)} GitHub repositories total.")
      return {"repositories": repositories}

  def fetch_full_papers_node(state: ResearchState) -> dict:
      downloaded = download_all_papers(state.get("papers", []))
      return {"downloaded_papers": downloaded}

  def index_research_papers_node(state: ResearchState) -> dict:
      """Convert, chunk, and index every downloaded research paper into Qdrant."""
      all_chunks = []

      for paper in state.get("downloaded_papers", []):
          try:
              markdown = convert_pdf(paper["local_path"])
          except Exception as exc:
              print(f"Failed to parse {paper['title']}: {exc}")
              continue

          chunks = chunk_markdown(
              markdown_text=markdown,
              paper_id=paper["arxiv_url"],
              paper_title=paper["title"],
          )
          all_chunks.extend(chunks)

      count = _upsert_chunks(all_chunks)
      return {"indexed_research": {"paper_chunks": count}, "paper_chunks": all_chunks}


  def _summarize_paper_for_critic(paper: dict, summary_chars: int = 220) -> dict:
      """Strip a paper dict down to only what the critic needs to judge relevance.
      Drops authors/categories entirely and truncates the abstract, since sending
      full abstracts for 10 papers on every loop iteration is what's blowing the
      Groq TPM budget."""
      summary = paper.get("summary", "") or ""
      if len(summary) > summary_chars:
          summary = summary[:summary_chars].rsplit(" ", 1)[0] + "..."
      return {
          "title": paper.get("title", ""),
          "summary": summary,
      }

  def critic_node(state: ResearchState) -> dict:
      loop_count = state.get("loop_count", 0) + 1

      prompt_template = ChatPromptTemplate.from_messages([
          ("system", CRITIC_PROMPT),
          ("user", "Evaluate the current search alignment for the query: {query}")
      ])

      trimmed_papers = [
          _summarize_paper_for_critic(p)
          for p in state.get("papers", [])[:5]
      ]

      messages = prompt_template.format_messages(
          problem_statement=state.get("markdown", ""),
          problem_summary=state.get("problem_summary", ""),
          papers=json.dumps(trimmed_papers, default=str),
          query=state["query"]
      )

      try:
          structured_llm = llm.with_structured_output(CriticResponse)
          parsed_response = structured_llm.invoke(messages)
          result = parsed_response.model_dump()          # build the dict first
      except Exception as e:
          print(f"Fallback caught parsing issue: {e}")
          result = {
              "relevance_score": 60,
              "feedback": "Parsing fallback triggered. Procedural fallback transition forward to retriever."
          }

      result["loop_count"] = loop_count                   # now safe in both paths
      return result

  def route_after_critic(state: ResearchState) -> str:
    if(state["relevance_score"] > 50) or state.get("loop_count", 0) >=2:
      return "retriever"
    else:
      return "orchestrator"


  def _tokenize(text: str) -> list[str]:
      return re.findall(r"\w+", text.lower())

  def build_bm25_index(chunks: list) -> tuple[BM25Okapi, list[dict]]:
      corpus_texts = [c.page_content for c in chunks]
      tokenized_corpus = [_tokenize(t) for t in corpus_texts]
      bm25 = BM25Okapi(tokenized_corpus)

      corpus_lookup = [
          {"text": c.page_content, **c.metadata}
          for c in chunks
      ]
      return bm25, corpus_lookup

  def dense_retrieve(query: str, top_k: int = 50, collection_name=COLLECTION_NAME):
    query_vector = embed_model.encode([query]).tolist()[0]
    hits = qdrant.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        query_filter=Filter(must=[FieldCondition(key="source_type", match=MatchValue(value="pdf"))]),
    ).points
    return [{"text": h.payload.get("text", ""), **h.payload, "dense_score": h.score} for h in hits]

  def reciprocal_rank_fusion(dense_results: list[dict], bm25_results: list[dict], k: int = 60, top_n: int = 50):
      """dense_results and bm25_results are lists of dicts sorted best-first, each with a unique 'chunk_id'."""
      scores = {}
      doc_lookup = {}

      for rank, doc in enumerate(dense_results):
          doc_id = doc["chunk_id"]
          scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
          doc_lookup[doc_id] = doc

      for rank, doc in enumerate(bm25_results):
          doc_id = doc["chunk_id"]
          scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
          doc_lookup[doc_id] = doc

      fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
      return [{"rrf_score": score, **doc_lookup[doc_id]} for doc_id, score in fused]

  def cross_encoder_rerank(query: str, candidates: list[dict], top_k: int = 10):
      pairs = [(query, c["text"]) for c in candidates]
      scores = cross_encoder.predict(pairs)   # returns a relevance score per pair, higher = more relevant

      for c, s in zip(candidates, scores):
          c["rerank_score"] = float(s)

      return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_k]

  def retrieve_node(state: ResearchState) -> dict:
      query = state["query"]
      chunks = state.get("paper_chunks", [])

      if not chunks:
          return {"retrieved_context": []}

      # Assumes each chunk has a stable chunk_id in metadata — add this in chunk_markdown if missing
      bm25, corpus_lookup = build_bm25_index(chunks)

      dense_hits = dense_retrieve(query, top_k=50)

      tokenized_query = _tokenize(query)
      bm25_scores = bm25.get_scores(tokenized_query)
      bm25_ranked = sorted(
          zip(corpus_lookup, bm25_scores), key=lambda x: x[1], reverse=True
      )[:50]
      bm25_hits = [{"bm25_score": score, **doc} for doc, score in bm25_ranked]

      fused = reciprocal_rank_fusion(dense_hits, bm25_hits, top_n=50)
      reranked = cross_encoder_rerank(query, fused, top_k=10)

      retrieved_context = [
          {
              "text": c["text"],
              "paper_title": c.get("paper_title", ""),
              "paper_id": c.get("paper_id", ""),
              "rerank_score": c["rerank_score"],
          }
          for c in reranked
      ]
      return {"retrieved_context": retrieved_context}


  def answer_node(state: ResearchState) -> dict:
      prompt_template = ChatPromptTemplate.from_messages([
          ("system", ANSWER_PROMPT),
          ("user", "{query}")
      ])

      messages = prompt_template.format_messages(
          problem_statement = problem_statement,  
          repositories=state["repositories"],
          retrieved_context = state["retrieved_context"],
          query=state["query"]
      )

      return {"answer" : llm.invoke(messages).content}


  graph = StateGraph(ResearchState)

  graph.add_node("parse_pdf", parse_pdf_node)
  graph.add_node("orchestrator", orchestrator_node)
  graph.add_node("arxiv_search", arxiv_search_node)
  graph.add_node("github_search", github_search_node)
  graph.add_node("fetch_full_papers",fetch_full_papers_node)
  graph.add_node("index_research_papers", index_research_papers_node)
  graph.add_node("critic",critic_node)
  graph.add_node("retriever", retrieve_node)
  graph.add_node("answer", answer_node)

  graph.add_edge(START, "parse_pdf")
  graph.add_edge("parse_pdf", "orchestrator")
  graph.add_edge("orchestrator", "arxiv_search")
  graph.add_edge("orchestrator", "github_search")
  graph.add_edge("arxiv_search", "fetch_full_papers")
  graph.add_edge("fetch_full_papers", "index_research_papers")
  graph.add_edge("index_research_papers", "critic")
  graph.add_edge("github_search", "critic")
  graph.add_conditional_edges(
      "critic",
      route_after_critic,
      {"retriever": "retriever", "orchestrator": "orchestrator"},
  )
  graph.add_edge("retriever", "answer")
  graph.add_edge("answer", END)

  app = graph.compile()



from google.colab import files
uploaded = files.upload()
pdf_path = list(uploaded.keys())[0]

initial_state = {
    "query": "Generate a research-backed technical roadmap for solving this hackathon problem statement. Include MVP steps, advanced steps, relevant papers, relevant GitHub repositories, and gaps.",
    "pdf_path": "/content/ObserveAI_M5_TechMeet14.pdf",
    "paper_id": "problem_statement_v1",
    "paper_title": "Inter-IIT Problem Statement",
}

result = app.invoke(initial_state)
print(result.get("answer"))