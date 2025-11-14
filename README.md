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