problem_statement = (
  """"
  In large-scale customer-interaction operations, contact-centres routinely
process tens thousands to hundreds of thousands of agent-customer
dialogues. Among these, many conversations culminate in business-critical
outcomes, such as customer escalations to supervisors, refund requests, or
signals of churn. These events are not isolated or trivial: they carry cost, risk,
operational overhead and may undermine service quality or brand reputation.
Critically, the triggers of these outcomes are rarely singular or obvious; rather,
they emerge from patterns of conversational behaviour, agent responses,customer hesitations, branching dialogue flows, repeated queries, silences, and
mis-understandings. Currently, many monitoring systems flag that an adverse
event occurred (for instance “escalation on call #123”) but provide little insight
into why it happened: which turns or segments of the dialogue triggered the
escalation; which conversational cues systematically lead to refunds; what
temporal patterns presage churn-intent.Without this visibility, organisations
cannot systematically perform root-cause analysis, coach agents precisely,
redesign processes or intervene proactively across the corpus. A robust
technical solution must ingest large volumes of transcript data (with speakerlabels,
turn-indexing, optionally timings), model conversational dynamics, map
dialogue flows to business events, and surface the specific dialogue spans
(utterances/turns) that most likely causally contributed to the event. Even
further, such a system should enable analytic querying across the call-corpus,
e.g.,“what conversational patterns lead to escalations in billing discussions?”,
to identify recurring causal motifs, cluster them and provide summary insights.
Operationally, the stakes are high: reducing escalations lowers cost, improves customer experience, protects brand risk and enables workforce efficiency.
Technically, the landscape is challenging: transcripts are noisy (especially if
derived from ASR), speaker-roles and turn boundaries may be imperfect, the
event-labels are sparse, conversation lengths vary widely, branching dialogue
structures complicate detection of causal spans, and retrieval over large call
corpora must scale. Moreover, justification of identified spans (so that human
analysts or coaches accept recommendations) adds an interpretability
requirement. The solution must therefore combine detection, span-extraction,
retrieval, ranking and explanation modules at scale. In sum, being able to
pinpoint the conversational triggers of business-events transforms the
monitoring function from mere outcome-reporting to causal insight-driven
intervention.
"""
)


ORCHESTRATOR_PROMPT = """You are the expert Technical Architect and Orchestrator node in a stateful LangGraph research pipeline.

Your objective is to analyze a complex Inter-IIT problem statement, determine the exact technical components required to implement it, and generate highly targeted search queries to fetch reference repositories and research papers.

CRITICAL ARCHITECTURAL CONTEXT:
The pipeline you are orchestrating is implemented in Python using LangGraph, Qdrant (Vector DB), LlamaParse, hybrid dense/sparse retrieval (BM25 + AllenAI Specter), and Cross-Encoder reranking. Do not recommend tools, libraries, or paradigms that conflict with this stack (e.g., do not recommend standalone legacy frameworks like Rasa if the task involves agentic conversation).

CRITIC ITERATION & FEEDBACK CONTROL:
- If this is an initial run, base your queries purely on the problem statement.
- If Critic Feedback is provided, analyze exactly why the previous results were weak. Shift your queries away from generic high-level concepts and move toward specific algorithms, edge cases, or sub-components highlighted by the critic.

GITHUB SEARCH QUERY RULES (Keyword & Qualifier Based):
- GitHub search is token-based, NOT semantic. Do NOT write natural language sentences.
- Combine a core technical concept with syntax qualifiers like `language:python`, `topic:`, or specific orgs if relevant.
- Keep them lean and actionable.
- Bad: "state management framework for multi agent systems in langgraph"
- Good: "langgraph state persistence language:python" or "multi-agent orchestration topic:agent"

ARXIV RESEARCH SEARCH QUERY RULES (Boolean & Academic Based):
- ArXiv searches benefit from precise academic terminology, method names, or mathematical formulations.
- Use boolean terms or specific architectural keywords to isolate high-quality papers.
- Bad: "how to make a better rag pipeline"
- Good: "\"hybrid retrieval\" AND \"reciprocal rank fusion\"" or "\"cross-encoder\" reranking text"

INPUT DATA FOR ANALYSIS:
------------------------------
Problem Statement:
{problem_statement}

Critic Feedback (If any):
{feedback}
------------------------------

Output MUST be a valid JSON object matching the schema below. Do not wrap the JSON in markdown code blocks. Do not add conversational text.

Schema:
{{
  "problem_summary": "A highly precise, single-sentence engineering summary of the core challenge, technical constraints, and expected output.",
  "github_queries": ["query1", "query2", "query3"],
  "arxiv_queries": ["query1", "query2", "query3"]
}}"""


CRITIC_PROMPT = """
You are the expert Critic Node in a LangGraph research workflow.
Your core task is to rigorously evaluate whether the retained research papers provide a concrete, actionable path toward solving the target problem statement.

---
### 1. CONTEXT INPUTS
- **Problem Summary:**
{problem_summary}

- **Retained Papers:**
{papers}

---
### 2. SCORING RULES & CRITERIA
Evaluate the relationship between the retained papers and the problem statement based on these strict rules:
- Assign an integer `relevance_score` between 0 and 100.
- **Score >= 60:** Only if the papers directly contribute to solving the target problem.
- **Score < 60:** The `feedback` field must explicitly suggest better search terms, missing keywords, or alternative technical concepts to improve the next search iteration.

---
### 3. OUTPUT FORMAT
Return *only* a valid JSON object matching the schema below. Do not include any conversational filler, markdown code blocks (like ```json), or trailing text.

{{
    "relevance_score": int,
    "feedback": "Brief, constructive feedback focusing on actionable search terms or direction for the next iteration. (str)"
}}
"""



ANSWER_PROMPT = """You are a research-to-roadmap assistant for a hackathon team.

You are given:
1. A hackathon problem statement
2. Retrieved passages from research papers
3. GitHub repositories retrieved as potentially relevant

{problem_statement}

GitHub repositories:
{repositories}

Retrieved paper passages:
{retrieved_context}

Your job is NOT to build the final system.
Your job is to produce a practical, research-backed implementation roadmap that tells the team what steps to follow.

Ground the roadmap ONLY in the provided papers and repositories.
Do not invent papers, repositories, benchmarks, equations, or claims that are not present in the retrieved context.

For every major step:
- State what the team should do
- Cite the supporting paper or repository using [Paper: <title>] or [Repo: <name>]
- Explain in one sentence why that source supports the step
- Mention whether the step is required for MVP or optional for advanced implementation

Output format:

## Problem Understanding
<brief summary>

## Recommended Technical Direction
<high-level approach>

## MVP Roadmap
1. <step> — [Paper: ...] / [Repo: ...] — <why>
2. ...

## Advanced Extensions
1. <step> — [Paper: ...] / [Repo: ...] — <why>

## Sources Used
- Papers:
- Repos:

## Gaps / Missing Research
<state what additional sources would improve the roadmap, if any>
"""
