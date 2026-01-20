#!/usr/bin/env python3
"""
Agentic Knowledge Graph Construction Pipeline

End-to-end pipeline for building knowledge graphs from both structured (CSV)
and unstructured (markdown/text) data using Google ADK agents.

Usage:
    # Interactive mode (default)
    python pipeline.py

    # With a specific query
    python pipeline.py --query "Build a supply chain BOM graph"

    # Non-interactive mode (auto-approve all stages)
    python pipeline.py --auto-approve --query "Build a supply chain BOM graph"

    # Verbose mode
    python pipeline.py --verbose --query "Build a supply chain graph"
"""

import os
import sys
import asyncio
import argparse
import logging
import warnings
from pathlib import Path
from itertools import islice
from typing import Optional, Dict, Any, List

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Google ADK imports
try:
    from google.adk.agents import Agent, LlmAgent, LoopAgent
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.tools import ToolContext
    from google.adk.sessions import InMemorySessionService
    from google.adk.runners import Runner
    from google.genai import types
except ImportError as e:
    logger.error(f"Failed to import Google ADK: {e}")
    logger.error("Install with: pip install google-adk")
    sys.exit(1)

# Local imports
from neo4j_for_adk import graphdb, tool_success, tool_error
from helper import get_neo4j_import_dir


# =============================================================================
# Constants
# =============================================================================
DEFAULT_MODEL = "openai/gpt-4o"

# State keys
PERCEIVED_USER_GOAL = "perceived_user_goal"
APPROVED_USER_GOAL = "approved_user_goal"
ALL_AVAILABLE_FILES = "all_available_files"
SUGGESTED_FILES = "suggested_files"
APPROVED_FILES = "approved_files"
PROPOSED_CONSTRUCTION_PLAN = "proposed_construction_plan"
APPROVED_CONSTRUCTION_PLAN = "approved_construction_plan"
PROPOSED_ENTITIES = "proposed_entity_types"
APPROVED_ENTITIES = "approved_entity_types"
PROPOSED_FACTS = "proposed_fact_types"
APPROVED_FACTS = "approved_fact_types"


# =============================================================================
# Tool Functions
# =============================================================================

# --- User Intent Tools ---

def set_perceived_user_goal(kind_of_graph: str, graph_description: str, tool_context: ToolContext):
    """Sets the perceived user's goal, including the kind of graph and its description.

    Args:
        kind_of_graph: 2-3 word definition of the kind of graph
        graph_description: a single paragraph description of the graph
    """
    user_goal_data = {"kind_of_graph": kind_of_graph, "graph_description": graph_description}
    tool_context.state[PERCEIVED_USER_GOAL] = user_goal_data
    return tool_success(PERCEIVED_USER_GOAL, user_goal_data)


def approve_perceived_user_goal(tool_context: ToolContext):
    """Upon approval from user, records the perceived user goal as the approved user goal.

    Only call this tool if the user has explicitly approved the perceived user goal.
    """
    if PERCEIVED_USER_GOAL not in tool_context.state:
        return tool_error("perceived_user_goal not set. Set perceived user goal first.")
    tool_context.state[APPROVED_USER_GOAL] = tool_context.state[PERCEIVED_USER_GOAL]
    return tool_success(APPROVED_USER_GOAL, tool_context.state[APPROVED_USER_GOAL])


def get_approved_user_goal(tool_context: ToolContext):
    """Returns the user's goal containing the kind of graph and its description."""
    if APPROVED_USER_GOAL not in tool_context.state:
        return tool_error("approved_user_goal not set. Ask the user to clarify their goal.")
    return tool_success(APPROVED_USER_GOAL, tool_context.state[APPROVED_USER_GOAL])


# --- File Suggestion Tools ---

def list_available_files(tool_context: ToolContext) -> dict:
    """Lists files available for knowledge graph construction."""
    import_dir_path = get_neo4j_import_dir()
    if not import_dir_path:
        # Fallback to local data directory
        import_dir_path = str(Path.cwd() / "data")

    import_dir = Path(import_dir_path)
    if not import_dir.exists() or not import_dir.is_dir():
        return tool_error(f"Import directory '{import_dir}' does not exist.")

    file_names = [
        str(x.relative_to(import_dir))
        for x in import_dir.rglob("*")
        if x.is_file()
    ]
    tool_context.state[ALL_AVAILABLE_FILES] = file_names
    return tool_success(ALL_AVAILABLE_FILES, file_names)


def sample_file(file_path: str, tool_context: Optional[ToolContext] = None) -> dict:
    """Samples a file by reading its content as text (up to 100 lines).

    Args:
        file_path: file to sample, relative to the import directory
    """
    import_dir_path = get_neo4j_import_dir()
    if not import_dir_path:
        import_dir_path = str(Path.cwd() / "data")

    import_dir = Path(import_dir_path)
    if not import_dir.exists():
        return tool_error(f"Import directory not found: {import_dir}")

    full_path = import_dir / file_path
    if not full_path.exists():
        return tool_error(f"File does not exist: {file_path}")

    try:
        with open(full_path, 'r', encoding='utf-8') as file:
            lines = list(islice(file, 100))
            content = ''.join(lines)
            return tool_success("content", content)
    except Exception as e:
        return tool_error(f"Error reading file {file_path}: {e}")


def set_suggested_files(suggest_files: List[str], tool_context: ToolContext) -> Dict[str, Any]:
    """Set the suggested files to be used for data import."""
    tool_context.state[SUGGESTED_FILES] = suggest_files
    return tool_success(SUGGESTED_FILES, suggest_files)


def get_suggested_files(tool_context: ToolContext) -> Dict[str, Any]:
    """Get the files suggested for data import."""
    return tool_success(SUGGESTED_FILES, tool_context.state.get(SUGGESTED_FILES, []))


def approve_suggested_files(tool_context: ToolContext) -> Dict[str, Any]:
    """Approves the suggested files for further processing."""
    if SUGGESTED_FILES not in tool_context.state:
        return tool_error("No suggested files to approve.")
    tool_context.state[APPROVED_FILES] = tool_context.state[SUGGESTED_FILES]
    return tool_success(APPROVED_FILES, tool_context.state[APPROVED_FILES])


def get_approved_files(tool_context: ToolContext):
    """Returns the files that have been approved for import."""
    if APPROVED_FILES not in tool_context.state:
        return tool_error("approved_files not set.")
    return tool_success(APPROVED_FILES, tool_context.state[APPROVED_FILES])


# --- Schema Proposal Tools (Structured) ---

def search_file(file_path: str, query: str) -> dict:
    """Searches a text file for lines containing the given query string.

    Args:
        file_path: Path to the file, relative to the import directory.
        query: The string to search for (case insensitive).
    """
    import_dir_path = get_neo4j_import_dir()
    if not import_dir_path:
        import_dir_path = str(Path.cwd() / "data")

    import_dir = Path(import_dir_path)
    p = import_dir / file_path

    if not p.exists():
        return tool_error(f"File does not exist: {file_path}")
    if not p.is_file():
        return tool_error(f"Path is not a file: {file_path}")

    if not query:
        return tool_success("search_results", {
            "metadata": {"path": file_path, "query": query, "lines_found": 0},
            "matching_lines": []
        })

    matching_lines = []
    search_query = query.lower()

    try:
        with open(p, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file, 1):
                if search_query in line.lower():
                    matching_lines.append({"line_number": i, "content": line.strip()})
    except Exception as e:
        return tool_error(f"Error searching file {file_path}: {e}")

    return tool_success("search_results", {
        "metadata": {"path": file_path, "query": query, "lines_found": len(matching_lines)},
        "matching_lines": matching_lines
    })


def propose_node_construction(
    approved_file: str,
    proposed_label: str,
    unique_column_name: str,
    proposed_properties: list[str],
    tool_context: ToolContext
) -> dict:
    """Propose a node construction for an approved file.

    Args:
        approved_file: The approved file to propose a node construction for
        proposed_label: The proposed label for constructed nodes
        unique_column_name: Column that uniquely identifies nodes
        proposed_properties: Column names to import as node properties
    """
    search_results = search_file(approved_file, unique_column_name)
    if search_results["status"] == "error":
        return search_results
    if search_results["search_results"]["metadata"]["lines_found"] == 0:
        return tool_error(f"{approved_file} does not have column {unique_column_name}.")

    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})
    node_construction_rule = {
        "construction_type": "node",
        "source_file": approved_file,
        "label": proposed_label,
        "unique_column_name": unique_column_name,
        "properties": proposed_properties
    }
    construction_plan[proposed_label] = node_construction_rule
    tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan
    return tool_success("node_construction", node_construction_rule)


def propose_relationship_construction(
    approved_file: str,
    proposed_relationship_type: str,
    from_node_label: str,
    from_node_column: str,
    to_node_label: str,
    to_node_column: str,
    proposed_properties: list[str],
    tool_context: ToolContext
) -> dict:
    """Propose a relationship construction for an approved file.

    Args:
        approved_file: The approved file
        proposed_relationship_type: The proposed relationship type
        from_node_label: The label of the source node
        from_node_column: Column identifying source nodes
        to_node_label: The label of the target node
        to_node_column: Column identifying target nodes
        proposed_properties: Properties to include on the relationship
    """
    search_results = search_file(approved_file, from_node_column)
    if search_results["status"] == "error":
        return search_results
    if search_results["search_results"]["metadata"]["lines_found"] == 0:
        return tool_error(f"{approved_file} does not have from_node_column {from_node_column}.")

    search_results = search_file(approved_file, to_node_column)
    if search_results["status"] == "error":
        return search_results
    if search_results["search_results"]["metadata"]["lines_found"] == 0:
        return tool_error(f"{approved_file} does not have to_node_column {to_node_column}.")

    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})
    relationship_construction_rule = {
        "construction_type": "relationship",
        "source_file": approved_file,
        "relationship_type": proposed_relationship_type,
        "from_node_label": from_node_label,
        "from_node_column": from_node_column,
        "to_node_label": to_node_label,
        "to_node_column": to_node_column,
        "properties": proposed_properties
    }
    construction_plan[proposed_relationship_type] = relationship_construction_rule
    tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan
    return tool_success("relationship_construction", relationship_construction_rule)


def remove_node_construction(node_label: str, tool_context: ToolContext) -> dict:
    """Remove a node construction from the proposed construction plan."""
    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})
    if node_label not in construction_plan:
        return tool_success("node_construction_removed", "not found, removal not needed")
    del construction_plan[node_label]
    tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan
    return tool_success("node_construction_removed", node_label)


def remove_relationship_construction(relationship_type: str, tool_context: ToolContext) -> dict:
    """Remove a relationship construction from the proposed construction plan."""
    construction_plan = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})
    if relationship_type not in construction_plan:
        return tool_success("relationship_construction_removed", "not found, removal not needed")
    construction_plan.pop(relationship_type)
    tool_context.state[PROPOSED_CONSTRUCTION_PLAN] = construction_plan
    return tool_success("relationship_construction_removed", relationship_type)


def get_proposed_construction_plan(tool_context: ToolContext) -> dict:
    """Get the proposed construction plan."""
    return tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN, {})


def approve_proposed_construction_plan(tool_context: ToolContext) -> dict:
    """Approve the proposed construction plan."""
    if PROPOSED_CONSTRUCTION_PLAN not in tool_context.state:
        return tool_error("No proposed construction plan found.")
    tool_context.state[APPROVED_CONSTRUCTION_PLAN] = tool_context.state.get(PROPOSED_CONSTRUCTION_PLAN)
    return tool_success(APPROVED_CONSTRUCTION_PLAN, tool_context.state[APPROVED_CONSTRUCTION_PLAN])


# --- Schema Proposal Tools (Unstructured) ---

def get_well_known_types(tool_context: ToolContext) -> dict:
    """Gets the approved labels that represent well-known entity types."""
    construction_plan = tool_context.state.get(APPROVED_CONSTRUCTION_PLAN, {})
    approved_labels = {
        entry["label"]
        for entry in construction_plan.values()
        if entry.get("construction_type") == "node"
    }
    return tool_success("approved_labels", list(approved_labels))


def set_proposed_entities(proposed_entity_types: list[str], tool_context: ToolContext) -> dict:
    """Sets the list of proposed entity types to extract from unstructured text."""
    tool_context.state[PROPOSED_ENTITIES] = proposed_entity_types
    return tool_success(PROPOSED_ENTITIES, proposed_entity_types)


def get_proposed_entities(tool_context: ToolContext) -> dict:
    """Gets the list of proposed entity types."""
    return tool_success(PROPOSED_ENTITIES, tool_context.state.get(PROPOSED_ENTITIES, []))


def approve_proposed_entities(tool_context: ToolContext) -> dict:
    """Approves the proposed entity types."""
    if PROPOSED_ENTITIES not in tool_context.state:
        return tool_error("No proposed entity types to approve.")
    tool_context.state[APPROVED_ENTITIES] = tool_context.state.get(PROPOSED_ENTITIES)
    return tool_success(APPROVED_ENTITIES, tool_context.state[APPROVED_ENTITIES])


def get_approved_entities(tool_context: ToolContext) -> dict:
    """Get the approved list of entity types."""
    return tool_success(APPROVED_ENTITIES, tool_context.state.get(APPROVED_ENTITIES, []))


def add_proposed_fact(
    approved_subject_label: str,
    proposed_predicate_label: str,
    approved_object_label: str,
    tool_context: ToolContext
) -> dict:
    """Add a proposed type of fact that could be extracted from the files.

    Args:
        approved_subject_label: approved label of the subject entity type
        proposed_predicate_label: label of the predicate
        approved_object_label: approved label of the object entity type
    """
    approved_entities = tool_context.state.get(APPROVED_ENTITIES, [])
    if approved_subject_label not in approved_entities:
        return tool_error(f"Subject label {approved_subject_label} not approved.")
    if approved_object_label not in approved_entities:
        return tool_error(f"Object label {approved_object_label} not approved.")

    current_predicates = tool_context.state.get(PROPOSED_FACTS, {})
    current_predicates[proposed_predicate_label] = {
        "subject_label": approved_subject_label,
        "predicate_label": proposed_predicate_label,
        "object_label": approved_object_label
    }
    tool_context.state[PROPOSED_FACTS] = current_predicates
    return tool_success(PROPOSED_FACTS, current_predicates)


def get_proposed_facts(tool_context: ToolContext) -> dict:
    """Get the proposed types of facts."""
    return tool_success(PROPOSED_FACTS, tool_context.state.get(PROPOSED_FACTS, {}))


def approve_proposed_facts(tool_context: ToolContext) -> dict:
    """Approves the proposed fact types."""
    if PROPOSED_FACTS not in tool_context.state:
        return tool_error("No proposed fact types to approve.")
    tool_context.state[APPROVED_FACTS] = tool_context.state.get(PROPOSED_FACTS)
    return tool_success(APPROVED_FACTS, tool_context.state[APPROVED_FACTS])


# =============================================================================
# Agent Definitions
# =============================================================================

def create_user_intent_agent(llm: LiteLlm) -> Agent:
    """Create the user intent agent."""
    instruction = """
    You are an expert at knowledge graph use cases.
    Your primary goal is to help the user come up with a knowledge graph use case.

    If the user is unsure what to do, make suggestions based on classic use cases like:
    - social network involving friends, family, or professional relationships
    - logistics network with suppliers, customers, and partners
    - recommendation system with customers, products, and purchase patterns
    - fraud detection over multiple accounts
    - pop-culture graphs with movies, books, or music

    A user goal has two components:
    - kind_of_graph: at most 3 words describing the graph
    - description: a few sentences about the intention of the graph

    Think carefully and collaborate with the user:
    1. Understand the user's goal
    2. Ask clarifying questions as needed
    3. Use 'set_perceived_user_goal' to record your perception
    4. Present the perceived user goal to the user for confirmation
    5. If the user agrees, use 'approve_perceived_user_goal' to approve it
    """

    return Agent(
        name="user_intent_agent",
        model=llm,
        description="Helps the user ideate on a knowledge graph use case.",
        instruction=instruction,
        tools=[set_perceived_user_goal, approve_perceived_user_goal]
    )


def create_file_suggestion_agent(llm: LiteLlm) -> Agent:
    """Create the file suggestion agent."""
    instruction = """
    You are a constructive critic AI reviewing a list of files. Your goal is to suggest
    relevant files for constructing a knowledge graph.

    Review the file list for relevance to the kind of graph and description specified
    in the approved user goal.

    For any file that you're not sure about, use the 'sample_file' tool to get
    a better understanding of the file contents.

    Consider both structured data files (CSV, JSON) and unstructured files (markdown, txt).

    Prepare for the task:
    - use 'get_approved_user_goal' to get the approved user goal

    Think carefully, repeating these steps until finished:
    1. List available files using 'list_available_files'
    2. Evaluate the relevance of each file
    3. Record suggested files using 'set_suggested_files'
    4. Use 'get_suggested_files' to retrieve and present them
    5. Ask the user to approve the set of suggested files
    6. If approved, use 'approve_suggested_files' to record the approval
    """

    return Agent(
        name="file_suggestion_agent",
        model=llm,
        description="Helps the user select files to import.",
        instruction=instruction,
        tools=[
            get_approved_user_goal, list_available_files, sample_file,
            set_suggested_files, get_suggested_files, approve_suggested_files
        ]
    )


def create_structured_schema_proposal_agent(llm: LiteLlm) -> Agent:
    """Create the schema proposal agent for structured (CSV) data."""
    instruction = """
    You are an expert at knowledge graph modeling with property graphs. Propose an
    appropriate schema by specifying construction rules which transform approved files
    into nodes or relationships.

    Consider feedback if it is available:
    <feedback>
    {feedback}
    </feedback>

    General guidance for identifying a node or a relationship:
    - If the file name is singular and has only 1 unique identifier, it is likely a node
    - If the file name combines two things, it is likely a full relationship
    - If a node file has multiple unique identifiers, some may be reference relationships

    Design rules for nodes:
    - Nodes will have unique identifiers
    - Nodes may have identifiers used as reference relationships

    Design rules for relationships:
    - Full relationships appear in dedicated relationship files
    - Reference relationships appear as foreign key references in node files

    Prepare for the task:
    - get user goal using 'get_approved_user_goal'
    - get approved files using 'get_approved_files'
    - get current construction plan using 'get_proposed_construction_plan'

    Think carefully:
    1. For each approved file, consider whether it represents a node or relationship
    2. Check content for potential unique identifiers using 'sample_file'
    3. For each identifier, verify uniqueness using 'search_file'
    4. For a node file, propose using 'propose_node_construction'
    5. For a relationship file, propose using 'propose_relationship_construction'
    6. When done, use 'get_proposed_construction_plan' to present the plan
    """

    return Agent(
        name="structured_schema_proposal_agent",
        model=llm,
        description="Proposes a knowledge graph schema based on structured data files",
        instruction=instruction,
        tools=[
            get_approved_user_goal, get_approved_files,
            get_proposed_construction_plan,
            sample_file, search_file,
            propose_node_construction, propose_relationship_construction,
            remove_node_construction, remove_relationship_construction,
            approve_proposed_construction_plan
        ]
    )


def create_ner_agent(llm: LiteLlm) -> Agent:
    """Create the NER (Named Entity Recognition) agent for unstructured data."""
    instruction = """
    You are a top-tier algorithm designer for analyzing text files and proposing
    the kind of named entities that could be extracted which would be relevant
    for a user's goal.

    Entities are people, places, things and qualities, but not quantities.
    Your goal is to propose the type of entities, not actual instances.

    Two approaches to identifying types of entities:
    - well-known entities: correlate with approved node labels in existing graph schema
    - discovered entities: appear consistently in the source text

    Design rules for well-known entities:
    - always use existing well-known entity types when possible
    - prefer reusing existing entity types rather than creating new ones

    Design rules for discovered entities:
    - discovered entities are consistently mentioned and relevant to user's goal
    - look for entities that provide more depth or breadth to the existing graph
    - avoid quantitative types (better as properties on existing entities)

    Prepare for the task:
    - use 'get_approved_user_goal' to get the user goal
    - use 'get_approved_files' to get the list of approved files
    - use 'get_well_known_types' to get the approved node labels

    Think step by step:
    1. Sample files using 'sample_file' to understand the content
    2. Consider well-known entities mentioned in the text
    3. Discover entities that are frequently mentioned
    4. Use 'set_proposed_entities' to save the list
    5. Use 'get_proposed_entities' to present them for approval
    6. If approved, use 'approve_proposed_entities' to finalize
    """

    return Agent(
        name="ner_schema_agent",
        model=llm,
        description="Proposes named entity types to extract from text files.",
        instruction=instruction,
        tools=[
            get_approved_user_goal, get_approved_files, sample_file,
            get_well_known_types,
            set_proposed_entities, get_proposed_entities, approve_proposed_entities
        ]
    )


def create_fact_agent(llm: LiteLlm) -> Agent:
    """Create the fact extraction agent for unstructured data."""
    instruction = """
    You are a top-tier algorithm designed for analyzing text files and proposing
    the type of facts that could be extracted that would be relevant for a user's goal.

    Do not propose specific individual facts, but the general type of facts.
    For example, not "ABK likes coffee" but "Person likes Beverage".

    Facts are triplets of (subject, predicate, object) where subject and object are
    approved entity types, and the predicate describes how they are related.

    Design rules for facts:
    - only use approved entity types as subjects or objects
    - the predicate should describe the relationship between subject and object
    - the predicate should optimize for information relevant to user's goal
    - the predicate must appear in the source text

    Prepare for the task:
    - use 'get_approved_user_goal' to get the user goal
    - use 'get_approved_files' to get the list of approved files
    - use 'get_approved_entities' to get the list of approved entity types

    Think step by step:
    1. Sample files using 'sample_file' to understand the content
    2. Consider how subjects and objects are related in the text
    3. Call 'add_proposed_fact' for each type of fact you propose
    4. Use 'get_proposed_facts' to retrieve all proposed facts
    5. Present the proposed types of facts to the user
    6. If approved, use 'approve_proposed_facts' to finalize
    """

    return Agent(
        name="fact_extraction_agent",
        model=llm,
        description="Proposes relevant fact types to extract from text files.",
        instruction=instruction,
        tools=[
            get_approved_user_goal, get_approved_files,
            get_approved_entities,
            sample_file,
            add_proposed_fact, get_proposed_facts, approve_proposed_facts
        ]
    )


# =============================================================================
# Agent Caller Utility
# =============================================================================

class AgentCaller:
    """A wrapper class for interacting with an ADK agent."""

    def __init__(self, agent: Agent, runner: Runner, user_id: str, session_id: str):
        self.agent = agent
        self.runner = runner
        self.user_id = user_id
        self.session_id = session_id

    async def get_session(self):
        return await self.runner.session_service.get_session(
            app_name=self.runner.app_name,
            user_id=self.user_id,
            session_id=self.session_id
        )

    async def call(self, query: str, verbose: bool = False) -> str:
        """Call the agent with a query and return the response."""
        print(f"\n>>> User: {query}")

        content = types.Content(role='user', parts=[types.Part(text=query)])
        final_response_text = "Agent did not produce a final response."

        async for event in self.runner.run_async(
            user_id=self.user_id,
            session_id=self.session_id,
            new_message=content
        ):
            if verbose:
                print(f"  [Event] Author: {event.author}, Final: {event.is_final_response()}")

            if event.is_final_response():
                if event.content and event.content.parts:
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate:
                    final_response_text = f"Agent escalated: {event.error_message or 'No message.'}"
                if event.author == self.agent.name:
                    break

        print(f"<<< Agent: {final_response_text[:500]}{'...' if len(final_response_text) > 500 else ''}")
        return final_response_text


async def make_agent_caller(
    agent: Agent,
    initial_state: Optional[Dict[str, Any]] = None
) -> AgentCaller:
    """Create and return an AgentCaller instance for the given agent."""
    if initial_state is None:
        initial_state = {}

    session_service = InMemorySessionService()
    app_name = agent.name + "_app"
    user_id = agent.name + "_user"
    session_id = agent.name + "_session"

    await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        state=initial_state
    )

    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=session_service
    )

    return AgentCaller(agent, runner, user_id, session_id)


# =============================================================================
# Pipeline Class
# =============================================================================

class KnowledgeGraphPipeline:
    """End-to-end pipeline for agentic knowledge graph construction."""

    def __init__(self, model: str = DEFAULT_MODEL, verbose: bool = False):
        """Initialize the pipeline.

        Args:
            model: The LLM model to use (default: openai/gpt-4o)
            verbose: Whether to print verbose output
        """
        self.llm = LiteLlm(model=model)
        self.verbose = verbose
        self.state: Dict[str, Any] = {}

    async def run(self, user_query: str, interactive: bool = True) -> Dict[str, Any]:
        """Run the complete pipeline.

        Args:
            user_query: Initial user query describing the graph they want to build
            interactive: Whether to prompt for user approval at each stage

        Returns:
            The final pipeline state containing all approved artifacts
        """
        print("\n" + "=" * 60)
        print("   AGENTIC KNOWLEDGE GRAPH CONSTRUCTION PIPELINE")
        print("=" * 60)

        # Stage 1: Clarify user intent
        await self.run_user_intent_stage(user_query, interactive)

        # Stage 2: Suggest and approve files
        await self.run_file_suggestion_stage(interactive)

        # Stage 3: Determine data type and propose schema
        await self.run_schema_proposal_stage(interactive)

        # Stage 4: Show summary
        self.print_summary()

        return self.state

    async def run_user_intent_stage(self, user_query: str, interactive: bool = True):
        """Stage 1: Clarify user intent and approve the goal."""
        print("\n" + "-" * 40)
        print("STAGE 1: User Intent Clarification")
        print("-" * 40)

        agent = create_user_intent_agent(self.llm)
        caller = await make_agent_caller(agent)

        # Initial call with user's query
        await caller.call(user_query, self.verbose)

        session = await caller.get_session()

        # Check if goal was set
        if PERCEIVED_USER_GOAL not in session.state:
            # Agent may need more information
            if interactive:
                follow_up = input("\nProvide more details (or 'skip' to auto-approve): ").strip()
                if follow_up.lower() != 'skip':
                    await caller.call(follow_up, self.verbose)
                    session = await caller.get_session()

        # Approve the goal
        if PERCEIVED_USER_GOAL in session.state and APPROVED_USER_GOAL not in session.state:
            if interactive:
                print(f"\nPerceived goal: {session.state[PERCEIVED_USER_GOAL]}")
                approval = input("Approve this goal? (y/n): ").strip().lower()
                if approval == 'y':
                    await caller.call("Approve that goal.", self.verbose)
                else:
                    feedback = input("Provide feedback: ").strip()
                    await caller.call(feedback, self.verbose)
                    await caller.call("Approve that goal.", self.verbose)
            else:
                await caller.call("Approve that goal.", self.verbose)

        session = await caller.get_session()
        self.state.update(session.state)

        if APPROVED_USER_GOAL in self.state:
            print(f"\n[OK] Approved goal: {self.state[APPROVED_USER_GOAL]}")
        else:
            print("\n[WARNING] No approved goal set")

    async def run_file_suggestion_stage(self, interactive: bool = True):
        """Stage 2: Suggest and approve files for import."""
        print("\n" + "-" * 40)
        print("STAGE 2: File Suggestion")
        print("-" * 40)

        agent = create_file_suggestion_agent(self.llm)
        caller = await make_agent_caller(agent, self.state.copy())

        # Ask agent to suggest files
        await caller.call("What files can we use for import?", self.verbose)

        session = await caller.get_session()

        if SUGGESTED_FILES in session.state:
            if interactive:
                print(f"\nSuggested files: {session.state[SUGGESTED_FILES]}")
                approval = input("Approve these files? (y/n): ").strip().lower()
                if approval == 'y':
                    await caller.call("Yes, approve those files.", self.verbose)
                else:
                    feedback = input("Provide feedback: ").strip()
                    await caller.call(feedback, self.verbose)
                    await caller.call("Approve those files.", self.verbose)
            else:
                await caller.call("Yes, approve those files.", self.verbose)

        session = await caller.get_session()
        self.state.update(session.state)

        if APPROVED_FILES in self.state:
            print(f"\n[OK] Approved files: {self.state[APPROVED_FILES]}")
        else:
            print("\n[WARNING] No approved files set")

    async def run_schema_proposal_stage(self, interactive: bool = True):
        """Stage 3: Propose schema based on data type."""
        print("\n" + "-" * 40)
        print("STAGE 3: Schema Proposal")
        print("-" * 40)

        approved_files = self.state.get(APPROVED_FILES, [])

        # Determine data types
        structured_files = [f for f in approved_files if f.endswith('.csv') or f.endswith('.json')]
        unstructured_files = [f for f in approved_files if f.endswith('.md') or f.endswith('.txt')]

        # Process structured files
        if structured_files:
            print(f"\nProcessing structured files: {structured_files}")
            await self.run_structured_schema_proposal(structured_files, interactive)

        # Process unstructured files
        if unstructured_files:
            print(f"\nProcessing unstructured files: {unstructured_files}")
            await self.run_unstructured_schema_proposal(unstructured_files, interactive)

    async def run_structured_schema_proposal(self, files: List[str], interactive: bool):
        """Propose schema for structured (CSV) data."""
        print("\n--- Structured Data Schema Proposal ---")

        # Update state with only structured files for this stage
        stage_state = self.state.copy()
        stage_state[APPROVED_FILES] = files
        stage_state["feedback"] = ""

        agent = create_structured_schema_proposal_agent(self.llm)
        caller = await make_agent_caller(agent, stage_state)

        await caller.call("How can these files be imported to construct the knowledge graph?", self.verbose)

        session = await caller.get_session()

        if PROPOSED_CONSTRUCTION_PLAN in session.state:
            if interactive:
                print(f"\nProposed construction plan:")
                for key, value in session.state[PROPOSED_CONSTRUCTION_PLAN].items():
                    print(f"  - {key}: {value['construction_type']}")

                approval = input("\nApprove this construction plan? (y/n): ").strip().lower()
                if approval == 'y':
                    await caller.call("Approve the proposed construction plan.", self.verbose)
                else:
                    feedback = input("Provide feedback: ").strip()
                    await caller.call(feedback, self.verbose)
                    await caller.call("Approve the proposed construction plan.", self.verbose)
            else:
                await caller.call("Approve the proposed construction plan.", self.verbose)

        session = await caller.get_session()
        self.state.update(session.state)

        if APPROVED_CONSTRUCTION_PLAN in self.state:
            print(f"\n[OK] Approved construction plan with {len(self.state[APPROVED_CONSTRUCTION_PLAN])} rules")

    async def run_unstructured_schema_proposal(self, files: List[str], interactive: bool):
        """Propose schema for unstructured (text/markdown) data."""
        print("\n--- Unstructured Data Schema Proposal ---")

        # Run NER agent first
        print("\nStep 3a: Named Entity Recognition")
        stage_state = self.state.copy()
        stage_state[APPROVED_FILES] = files

        ner_agent = create_ner_agent(self.llm)
        ner_caller = await make_agent_caller(ner_agent, stage_state)

        await ner_caller.call(
            "Propose entity types that can be extracted from these text files.",
            self.verbose
        )

        session = await ner_caller.get_session()

        if PROPOSED_ENTITIES in session.state:
            if interactive:
                print(f"\nProposed entities: {session.state[PROPOSED_ENTITIES]}")
                approval = input("Approve these entity types? (y/n): ").strip().lower()
                if approval == 'y':
                    await ner_caller.call("Approve the proposed entities.", self.verbose)
                else:
                    feedback = input("Provide feedback: ").strip()
                    await ner_caller.call(feedback, self.verbose)
                    await ner_caller.call("Approve the proposed entities.", self.verbose)
            else:
                await ner_caller.call("Approve the proposed entities.", self.verbose)

        session = await ner_caller.get_session()
        self.state.update(session.state)

        # Run Fact agent
        if APPROVED_ENTITIES in self.state:
            print("\nStep 3b: Fact Type Extraction")

            fact_agent = create_fact_agent(self.llm)
            fact_caller = await make_agent_caller(fact_agent, self.state.copy())

            await fact_caller.call(
                "Propose fact types that can be found in the text.",
                self.verbose
            )

            session = await fact_caller.get_session()

            if PROPOSED_FACTS in session.state:
                if interactive:
                    print(f"\nProposed fact types: {list(session.state[PROPOSED_FACTS].keys())}")
                    approval = input("Approve these fact types? (y/n): ").strip().lower()
                    if approval == 'y':
                        await fact_caller.call("Approve the proposed fact types.", self.verbose)
                    else:
                        feedback = input("Provide feedback: ").strip()
                        await fact_caller.call(feedback, self.verbose)
                        await fact_caller.call("Approve the proposed fact types.", self.verbose)
                else:
                    await fact_caller.call("Approve the proposed fact types.", self.verbose)

            session = await fact_caller.get_session()
            self.state.update(session.state)

        if APPROVED_ENTITIES in self.state:
            print(f"\n[OK] Approved entity types: {self.state[APPROVED_ENTITIES]}")
        if APPROVED_FACTS in self.state:
            print(f"[OK] Approved fact types: {list(self.state[APPROVED_FACTS].keys())}")

    def print_summary(self):
        """Print a summary of the pipeline results."""
        print("\n" + "=" * 60)
        print("   PIPELINE SUMMARY")
        print("=" * 60)

        if APPROVED_USER_GOAL in self.state:
            goal = self.state[APPROVED_USER_GOAL]
            print(f"\nGoal: {goal.get('kind_of_graph', 'N/A')}")
            print(f"Description: {goal.get('graph_description', 'N/A')[:200]}...")

        if APPROVED_FILES in self.state:
            print(f"\nApproved Files ({len(self.state[APPROVED_FILES])}):")
            for f in self.state[APPROVED_FILES]:
                print(f"  - {f}")

        if APPROVED_CONSTRUCTION_PLAN in self.state:
            plan = self.state[APPROVED_CONSTRUCTION_PLAN]
            nodes = [k for k, v in plan.items() if v.get('construction_type') == 'node']
            rels = [k for k, v in plan.items() if v.get('construction_type') == 'relationship']
            print(f"\nConstruction Plan:")
            print(f"  Nodes: {nodes}")
            print(f"  Relationships: {rels}")

        if APPROVED_ENTITIES in self.state:
            print(f"\nApproved Entity Types: {self.state[APPROVED_ENTITIES]}")

        if APPROVED_FACTS in self.state:
            print(f"Approved Fact Types: {list(self.state[APPROVED_FACTS].keys())}")

        print("\n" + "=" * 60)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Agentic Knowledge Graph Construction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py
  python pipeline.py --query "Build a supply chain BOM graph"
  python pipeline.py --auto-approve --query "Build a supply chain graph"
  python pipeline.py --verbose --query "Create a product review graph"
  python pipeline.py --model openai/gpt-4o --query "Build a social network"
        """
    )

    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Initial query describing the graph to build"
    )

    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve all stages (non-interactive mode)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model to use (default: {DEFAULT_MODEL})"
    )

    args = parser.parse_args()

    # Get query interactively if not provided
    if args.query is None:
        print("\nWelcome to the Agentic Knowledge Graph Construction Pipeline!")
        print("Describe the kind of knowledge graph you want to build.\n")
        args.query = input("Your query: ").strip()
        if not args.query:
            args.query = "Help me build a knowledge graph"

    # Create and run the pipeline
    pipeline = KnowledgeGraphPipeline(
        model=args.model,
        verbose=args.verbose
    )

    # Run the async pipeline
    asyncio.run(
        pipeline.run(
            user_query=args.query,
            interactive=not args.auto_approve
        )
    )


if __name__ == "__main__":
    main()
