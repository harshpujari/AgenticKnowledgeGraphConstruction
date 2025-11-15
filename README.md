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
1. Reflection
  In reflection workflows, the agent generates an initial output and then critically evaluates it, identifying flaws, inconsistencies, or areas for improvement. Based on this self-critique, the agent revises its output through multiple iterations until it meets quality standards.
  How it works: The agent alternates between generation and critique phases. In the critique phase, it examines its own work from different angles (accuracy, completeness, clarity, logic) and produces specific feedback. Then it uses that feedback to generate an improved version.
  
  Use cases:

    Code generation and debugging: The agent writes code, reviews it for bugs or inefficiencies, then refines it
    
    Essay writing: Drafting content and iteratively improving structure, argumentation, and clarity
    
    Data analysis: Generating analytical insights, checking for logical errors or overlooked patterns, then revising conclusions
    
    Creative content: Writing stories or marketing copy where quality improvement through self-editing is valuable