# CSC 581 Final Project: Repo Assist
### Arian Houshmand, C.J. DuHamel, Colin Ngo, Charlie Ray, Jason Jelincic


## Final Project Overview
Members: Arian Houshmand, C.J. DuHamel, Colin Ngo, Charlie Ray, Jason Jelincic

Repo Assist
Project Overview
The goal of this project is to develop an assistant for open-source software development that can reason over repository structure, code, and development history to support common developer workflows. We recognize that open-source repositories often include multiple sources of information beyond source code, such as commit histories, issues, and pull requests, which together provide important context for development decisions. The purpose of this project is to explore how a large language model, equipped with repository-level tools, can use this information to assist with tasks such as identifying next development steps, improving code clarity, and supporting documentation and maintenance. 
The intended users of this system are developers working in collaborative open-source environments, particularly those new to the codebases. As part of the project implementation, we plan to focus on repositories from Cal Poly’s Hack4Impact organization, which provides a consistent and well-documented development structure across multiple projects. Using these repositories allows us to study how an assistant can learn from established workflows, conventions, and historical artifacts such as issues and pull requests. This context enables evaluation of the assistant’s ability to generalize across similar projects while remaining grounded in real world open source development practices.

Background
In the past we have seen companies develop their own AI pair programers and integrated workspace agents, such as devon, claude code, cursor, windsurf, etc. This will be our own take with a unique goal based on those ideas. We might take advantage of MCP for tooling. Previous tools use MCP for tool calling, and can search the web for documentation, open new terminal commands, and browse through GitHub repository history. 

Related work:
https://github.com/entropy-research/Devon
https://github.com/ai-genie/chatgpt-vscode
https://github.com/ishaan1013/sandbox
https://github.com/CRJFisher/code-charter
https://github.com/ObjectJosh/continue.dev
https://github.com/hack4impact-calpoly
https://github.com/modelcontextprotocol/servers

Huggingface Dataset: missvector/linux-commands Nanbeige/ToolMind

Difficulty
9-10, depending on scope. Fine tuning would add more difficulty. If needed we can just focus on robust MCP functionality, to enable the models to browse and suggest improvements to existing repositories.

Relevance
This project aligns with the course topics of using Generative AI to support knowledge-intensive tasks. Large software repositories often contain many different types of knowledge, including source code, version history, issues, and pull requests. Having an AI agent to sort through this knowledge base and provide useful insights aligns with the course objectives. We will create an agentic workflow where the model uses tools to search files and reason over repository structure. 

Schedule

Week
Goals
4
Research into relevant work, such as other attempts at creating repository assistants, along with parsing strategies for complex repositories and documentation.
5
Begin dataset creation and exploratory data analysis, and begin developing a strategy for implementation, such as prompt strategies
6
Continue dataset creation and repository parsing, including GitHub specific actions such as CI/CD (tooling around software development workflows), begin processing data and structuring for LLM input, begin research into possible fine-tuning strategies. 
7
Research into UX/UI options, begin implementing LLM into backend with dataset
8
UX/UI Implementation, testing, and bugfixing backend
9
Include Fine Tuning for LLM + Bug Fixing
10
Fine Tuning Presentation


## Features, Requirements and Evaluation Criteria
Members: Arian Houshmand, C.J. DuHamel, Colin Ngo, Charlie Ray, Jason Jelincic
Features
	Features describe the system from an end-user perspective, we focus on usefulness, accessibility, and providing value to open source developers. Repo Assist functions as an intelligent agent that helps developers understand, navigate and contribute to unfamiliar open-source repositories. The following are our planned features:
Repo Assist will provide high level explanations of repository structure, including key files and their roles within the project. A user should be able to ask questions like “What does this repository do?” or “Where is the authentication implemented?” and receive accurate, context-aware responses.
By using historical artifacts such as pull requests and issues, Repo Assist can provide established development practices within Hack4Impact repositories. This should help new contributors align their changes with existing workflows and best practices.
The system will support a conversational interface that allows iterative questioning and refinement. Users will be able to ask follow up questions or request clarification on any statements made by the agent.
Repo Assist will be able to suggest documentation improvements, or suggest next development steps that would be most impactful/useful to the current repository.
Requirements
Repo Assist should be able to ingest a GitHub repository, including its file structure, source code, commit history, and pull requests.
Repo Assist will support natural language queries about repository structure, code functionality, and development history. The agent will provide accurate, and task-relevant answers.
Repo Assist will support tool use, being able to search for files, commit lookups, or gather recent issues, to retrieve relevant context before generating responses, rather than relying solely on pre-trained information.
Repo Assist will maintain conversational context and history with users, allowing follow up questions. The system shall allow users to reset or clear conversation history/context
Repo Assist shall suggest next development tasks and label them by impact (high/medium/low) and effort estimate. It shall generate documentation for next improvement containing targeted files, proposed change, justification of change tied to repo content.
Repo Assist will run as a web app or CLI accessible on modern web browsers
Evaluation Criteria
For a predefined set of questions (“Where is X implemented?”, “What issues are currently open?”, “Can you explain the role of X file?” “What issues are currently open?) the system’s responses will be evaluated for correctness against ground source truth from manual inspection.
The relevance of files, commits, issues, or pull requests selected by the system will be assessed. A response will be considered successful if it cites information directly related to the user’s query. An unsuccessful response would be one that cites information unrelated to the user's query. For example, if the user asks “where is authentication implemented” the agent should not provide information unrelated to authentication.
The system will be tested with multi-turn interactions to verify that it maintains and correctly uses prior conversational context when answering follow up questions. The system will also be evaluated on whether resetting the conversation clears prior context as expected.
Suggested next development tasks will be evaluated based on relevance to the repository’s current state. Recommendations will be considered successful if they reference specific files or components and provide a clear justification tied to repository content. Impact and effort labels will be checked for internal consistency and reasonable alignment with project scope
Repo Assist will be evaluated on whether it can be accessed as a web application or CLI through modern web browsers. The system will be considered usable if users can successfully submit queries, receive responses, and navigate follow-up interactions without errors.
When the system cannot answer a question due to missing information or ambiguity, it will be evaluated on whether it clearly communicates uncertainty rather than producing incorrect or fabricated information.
The system will be evaluated on its ability relative to other current solutions, including both performance comparison for common attributes and feature comparison (i.e. features that set our project apart from others)


## System Design and Architecture
Members: Arian Houshmand, C.J. DuHamel, Colin Ngo, Charlie Ray, Jason Jelincic
Repo Assist
Overview
Repo Assist is a tool-augmented large language model (LLM) agent that helps developers understand and contribute to unfamiliar open-source repositories. The system answers questions by retrieving evidence from repository components (files, line ranges, issues, pull requests) and only generates responses which are grounded in those components. Every factual claim about the repository is accompanied by a citation to a specific piece of the code.
Some features may include in contrast to preexisting solutions:
Viewing through Git History
Suggestions to create changes via a specific method based off history for specific flies
CLAUDE.ME File Type solution
Citations to lines of code

	At an algorithmic level, Repo Assist follows a planner-executor agent loop. It first interprets the users intent, then it plans a small sequence of retrieval actions, then executes tool calls to gather evidence, and finally synthesizes an evidence-backed answer (and occasionally a patch for code if a user requested it). The design prioritizes correctness, traceability, and controllable tool usage.

Description
Repo Assist operates on one repository at a time, provided as a local checkout path (the user pulls the repo locally before starting). The system performs an initial ingestion step to create an internal representation of the repositories structure (tree, file metadata, and chunk boundaries for line-based citations). For questions involving project management context (i.e. “Which issues should I prioritize?”), Repo Assist can also ingest issue and pull request metadata into the same evidence store.

When a user asks a question like “Where is authentication implemented?” the agent:
Classifies the request as a “location/feature finding intent”.
Uses repository search to identify candidate files (and related PRs or issues if relevant).
Opens specific files, and reads specific regions to confirm implementation details.
Produces an answer that references the retrieved information.

If Repo Assist can’t find sufficient evidence it responds with a clear limitation statement, i.e. “Unable to find feature X in Y files”

Functional Components
CLI Interface
Accepts repository context (local files) and natural-language requests. Presents answers, citations, and optional patches.
Inputs: repo_path, user_query, mode (optional: explain/locate/suggest/patch), scope (optional: files only vs include PR)
Outputs: answer_text, citations (list), patch_diff, next_actions (list)
Session Manager
Maintains conversational context and user preferences across conversations
Key state holds the current repository identifier, recent queries, selected evidence references, and user settings
Stores session information as a JSON object, so it can easily be passed into the LLM API
Agent Orchestrator
Converts user requests into tool plans and evidence-backed responses
Uses a planner-executer design pattern
Outputs: tool call plan, executed tool calls, consolidated evidence sets, final response.
Tool Gateway
Presents a uniform abstraction for evidence acquisition, independent of underlying integrations. This keeps the architecture stable even if tool implementations change (i.e. MCP, vs local tools, etc)
Core tool API’s
search_repo(query, filters) -> Evidence[]
open_file(path, start_line, end_line) -> Text
get_issue(query, state, labels, limits) -> Issue
get_pull_requests(query, state, labels, limit) -> PRs
Repository Ingestion & Indexer
Prepares repository code for retrieval and citation
Extracts file metadata, chunks files into line groups (with descriptions of the functions), creates lexical indexes (file X has Y), optionally can ingest PRs and open issues as well.
Retriever / Ranker
Produces a ranked list of candidate evidence items for each query
lexical retrieval over file paths and chunk text
heuristics for prioritization (e.g., README/docs first for “what does this repo do?”, likely module paths for “where is X implemented?”)
optional semantic retrieval as an extension (not required for the baseline design)
Evidence Store
Stores normalized objects for grounding and citation(s), basically just a long list of text along with metadata about where that text came from. This is so the agent can actually insert citations into what it states.
Response Composer
Produces a natural language response to the users query, with citations supporting each claim. We will put constraints on it to try not to claim anything without a citation directly from the code.
Diagram

System Overview:

Planner-Executor Agent Loop:

Data Model


To represent the codebase, we design a function to preprocess the codebase into a condensed version, enabling the entire codebase to fit within the context window of our model. To do this, we traverse the directory tree recursively, building out the leaf nodes (files) in the form of JSON objects that contain LLM-generated tags for each file. These JSON files can contain relevant metadata, such as relevant PRs, Issues, and git history. These JSON objects are then connected to parent nodes to represent files, which will then be assigned similar metadata, including additional LLM-generated tags based on the JSON objects in the leaf nodes. This should assist in queries like “where is authentication implemented?” as the LLM would be provided with detailed information about the full directory structure. In addition, having this directory structure allows for more primitive versions of search if necessary, including file retrieval for file summary if asked. 
PRs and Issues can also be represented as JSON objects and fed into the context. If this step is done first, we can then utilize these PRs and Issues to tag associated nodes in the directory structure. We would like the model to utilize the structure of this data to provide better insights with relevant context without requiring a very large context window, which will be particularly efficient for large code bases.
We plan to have the following objects within our JSON objects.
Repository: identification + indexing metadata
File: path, language/type, metadata
Chunk: a line-bounded segment of a file for citations
Issue: id, title/body, labels/state, timestamps
PullRequest: id, title/body, labels/state, timestamps, related changes (if available)
	With the following relationships:
Repository contains Files
File contains Chunks
Issue/PR references Files/Chunks (derived from retrieval + explicit citation use)
PR touches Files (when changed-file metadata is available)


The diagram below shows this visually:
 
