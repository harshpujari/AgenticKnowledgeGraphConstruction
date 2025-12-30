## My source to study: https://learn.deeplearning.ai/

# Agentic Knowledge Graph Construction

A knowledge graph is a structured representation of information that stores data as a network of interconnected entities and their relationships.

## Overview

Knowledge graphs model real-world information as nodes (entities) and edges (relationships). They enable machines to understand context and meaning beyond simple keyword matching, powering semantic search, recommendations, and higher-level AI reasoning.

## Key characteristics

- **Structure**
  - Nodes represent entities (people, places, concepts).
  - Edges represent relationships between entities, forming a graph database.

- **Format**
  - Commonly represented as subject–predicate–object triples.
  - Example: `"Paris" (subject) → "is capital of" (predicate) → "France" (object)`.

- **Purpose**
  - Capture context and semantic relationships so applications can reason about data rather than just match text.

## Common examples

- Google Knowledge Graph (enhances search results)
- Amazon product recommendation systems
- Wikidata (structured data underlying Wikipedia)

## Note on "graph"

The term "graph" refers to the mathematical graph structure (nodes and edges), not a visual chart or plot. It denotes a network of connected data points that can be queried and reasoned over.

## What is a lexical graph ?

- A lexical graph is a graph structure that represents relationships between words (lexemes) based on their linguistic and semantic connections.
Structure: Nodes are words/terms, edges represent lexical relationships like:

    - Synonymy (similar meaning)
    - Antonymy (opposite meaning)
    - Hypernymy (is-a: "dog" is a "mammal")
    - Hyponymy (specific type: "poodle" is a type of "dog")
    - Meronymy (part-of: "wheel" is part of "car")

Example using WordNet (the most famous lexical database):

    "vehicle" (hypernym - broader category)
        ↓
    "car" (base word)
        ↓
    "sedan", "SUV", "coupe" (hyponyms - specific types)
    "car" ← synonym → "automobile"
    "car" ← antonym → (no direct antonym)
    "car" ← meronym → "wheel", "engine", "door"

**Practical example:**
When you search "automobile repair" and a system understands to also show results for "car repair" or "vehicle maintenance," it's using a lexical graph to understand that these terms are related.

Difference from knowledge graph:
- Lexical graph: relationships between words themselves (linguistic)
- Knowledge graph: relationships between real-world entities (factual)

**What is an Agentic Workflow?**

An agentic workflow refers to a system where an AI agent autonomously executes tasks through a series of decision-making steps, rather than simply responding to a single prompt. Unlike traditional LLM interactions that follow a one-shot request-response pattern, agentic workflows allow the AI to take multiple actions, reason about intermediate results, use tools, and iteratively work toward completing complex objectives. The agent acts with agency—making decisions about what to do next based on context and goals.

Major Types of Agentic Workflows
1. **Reflection**
  In reflection workflows, the agent generates an initial output and then critically evaluates it, identifying flaws, inconsistencies, or areas for improvement. Based on this self-critique, the agent revises its output through multiple iterations until it meets quality standards.
  How it works: The agent alternates between generation and critique phases. In the critique phase, it examines its own work from different angles (accuracy, completeness, clarity, logic) and produces specific feedback. Then it uses that feedback to generate an improved version.
  
  Use cases:

    Code generation and debugging: The agent writes code, reviews it for bugs or inefficiencies, then refines it
    
    Essay writing: Drafting content and iteratively improving structure, argumentation, and clarity
    
    Data analysis: Generating analytical insights, checking for logical errors or overlooked patterns, then revising conclusions
    
    Creative content: Writing stories or marketing copy where quality improvement through self-editing is valuable

2. **Tool Use**
  Tool use workflows enable agents to interact with external systems, APIs, databases, or functions to accomplish tasks they cannot complete through language generation alone. The agent decides which tools to call, with what parameters, interprets the results, and incorporates them into its reasoning.
  How it works: The agent is provided with a set of available tools and their specifications. When faced with a task, it reasons about which tool(s) to use, makes function calls with appropriate arguments, receives structured results, and uses those results to continue its work or respond to the user.

  Use cases:

    Web research: Searching the internet, fetching web pages, and synthesizing information from multiple sources

    Data retrieval: Querying databases, pulling information from APIs, or accessing internal company systems

    Calculations: Using calculators, running code, or performing mathematical operations beyond the model's native capabilities

    File operations: Reading, writing, and manipulating documents, spreadsheets, or other files
    
    Integration tasks: Booking appointments, sending emails, updating CRM systems, or triggering workflows in external platforms

3. **Planning**
  Planning workflows involve decomposing complex, multi-step tasks into smaller subtasks, creating an execution plan, and then systematically working through that plan. The agent may create the entire plan upfront or adapt it dynamically as it progresses.
  How it works: Given a high-level goal, the agent first analyzes what needs to be accomplished and breaks it into logical steps or subgoals. It then executes these steps sequentially, tracking progress and potentially revising the plan based on intermediate results or obstacles encountered.

  Use cases:

    Project management: Breaking down a project into tasks, dependencies, and timelines

    Travel planning: Coordinating flights, hotels, activities, and logistics for a trip

    Research tasks: Structuring a comprehensive investigation into a topic with multiple research questions

    Software development: Planning feature implementation across multiple components and files
    
    Complex problem-solving: Mathematical proofs, strategic analysis, or multi-stage business decisions

4. **Multi-Agent Collaboration**
  Multi-agent workflows involve multiple AI agents working together, each potentially with different roles, expertise, or perspectives. Agents may debate, divide responsibilities, or sequentially refine each other's work.
  How it works: Different agent instances are assigned specific roles or personas. They interact through structured communication—one agent's output becomes another's input. This can take forms like debate (agents argue different positions), assembly line (each agent handles one stage), or collaborative (agents jointly solve problems with complementary skills).

  Use cases:

    Code review: One agent writes code, another reviews it, a third suggests improvements

    Content creation: Separate agents for research, writing, and editing

    Decision-making: Multiple agents representing different stakeholder perspectives to evaluate options

    Simulation and gaming: Agents playing different roles in simulated scenarios or game environments

    Complex analysis: Different agents specializing in different analytical frameworks or domains

5. **ReAct (Reasoning + Acting)**
  ReAct workflows create a tight loop where the agent alternates between thinking (reasoning about what to do) and acting (taking concrete steps). Each action's result informs the next reasoning step, creating an interleaved thought-action-observation cycle.
  How it works: The agent explicitly generates reasoning traces before each action, explaining its thought process and why it's choosing a particular action. After acting, it observes the result, reasons about what it learned, and decides the next action. This continues until the task is complete.
  
  Use cases:

    Interactive problem-solving: Navigating complex environments where each step reveals new information
    
    Debugging: Trying different fixes, observing results, reasoning about what worked or didn't

    Exploratory research: Searching for information where each discovery shapes the next query

    Dynamic task execution: Situations where the path isn't clear upfront and must be discovered through exploration

    Question answering: Complex queries requiring multiple information-gathering steps with reasoning between each

~~~
                    ╔════════════════════════════════╗
                    ║   Knowledge Graph Agent        ║
                    ╚══════════╦═════════════════════╝
                               ║
          ┌────────────────────┼────────────────────┐
          │                    │                    │
    ┌─────▼─────┐      ┌───────▼──────┐     ┌──────▼──────┐
    │Structured │      │Unstructured  │     │  GraphRAG   │
    │Data Agent │      │ Data Agent   │     │    Agent    │
    └─────┬─────┘      └───────┬──────┘     └──────┬──────┘
          │                    │                    │
    ┌─────┴─────┐        ┌─────┴─────┐             │
    │           │        │           │             │
┌───▼───┐   ┌───▼───┐┌───▼───┐   ┌───▼───┐     ┌──▼──┐
│[L4]   │   │[L5]   ││[L4]   │   │[L5]   │     │ ▄▄▄ │
│User   │   │File   ││User   │   │File   │     │█   █│
│Intent │   │Sugges-││Intent │   │Sugges-│     │█ K █│
│Agent  │   │tion   ││Agent  │   │tion   │     │█ n █│
└───┬───┘   │Agent  │└───┬───┘   │Agent  │     │█ o █│
    │       └───┬───┘    │       └───┬───┘     │█ w █│
    │   ┌───▼───┐        │   ┌───▼───┐         │█ l █│
    │   │[L6]   │        │   │[L7]   │         │█ e █│
    │   │Schema │        │   │Entity │         │█ d █│
    │   │Propo- │        │   │& Fact │         │█ g █│
    │   │sal    │        │   │Type   │         │█ e █│
    │   │Agent  │        │   │Propo- │         │█   █│
    │   └───┬───┘        │   │sal    │         │█ G █│
    │       │            │   │Agent  │         │█ r █│
    └───────┴────────────┘   └───┬───┘         │█ a █│
            │                    │             │█ p █│
            ▼                    ▼             │█ h █│
    ┌───────────────┐    ┌───────────────┐     │▀▀▀▀▀│
    │    Graph      │    │   Knowledge   │     └──▲──┘
    │ Construction  │───▶│  Extraction   │────────┘
    │     Plan      │    │     Plan      │
    └───────────────┘    └───────┬───────┘
                                 │
                         ┌───────▼───────────┐
                         │[L8] Knowledge     │
                         │Graph Construction │
                         │      Tool         │
                         └───────────────────┘
~~~

hi