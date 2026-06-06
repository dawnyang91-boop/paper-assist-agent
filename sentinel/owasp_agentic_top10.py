from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


ASI01 = "agent_goal_hijack"
ASI02 = "tool_misuse_exploitation"
ASI03 = "identity_privilege_abuse"
ASI04 = "agentic_supply_chain"
ASI05 = "unexpected_code_execution"
ASI06 = "memory_context_poisoning"
ASI07 = "insecure_inter_agent_communication"
ASI08 = "cascading_failures"
ASI09 = "human_agent_trust_exploitation"
ASI10 = "rogue_agents"


@dataclass(frozen=True)
class OwaspAgenticItem:
    id: str
    key: str
    name: str
    description: str
    covered_by_existing_modules: List[str]
    dataset_files: List[str]
    detector_modules: List[str]
    defense_modules: List[str]
    runtime_hooks: List[str]
    current_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


OWASP_AGENTIC_TOP10: Dict[str, OwaspAgenticItem] = {
    "ASI01": OwaspAgenticItem(
        id="ASI01",
        key=ASI01,
        name="Agent Goal Hijack",
        description="Attempts to override the agent's original goal, hierarchy, or safety instructions.",
        covered_by_existing_modules=["InputGuard", "RAGSanitizer", "ToolOutputSanitizer", "MemorySafetyChecker"],
        dataset_files=["prompt_injection.yaml", "jailbreak.yaml"],
        detector_modules=["PromptInjectionDetector", "JailbreakDetector"],
        defense_modules=["InputGuard", "RAGSanitizer", "ToolOutputSanitizer"],
        runtime_hooks=["user_input", "rag_retrieval", "tool_output", "memory_read"],
        current_status="strong",
    ),
    "ASI02": OwaspAgenticItem(
        id="ASI02",
        key=ASI02,
        name="Tool Misuse & Exploitation",
        description="Unsafe, unauthorized, destructive, or exfiltrating tool use.",
        covered_by_existing_modules=["ToolPolicyChecker", "ToolOutputSanitizer"],
        dataset_files=["tool_misuse.yaml"],
        detector_modules=["ToolPolicyChecker"],
        defense_modules=["ToolPolicyChecker", "ToolOutputSanitizer"],
        runtime_hooks=["tool_call", "tool_output"],
        current_status="strong",
    ),
    "ASI03": OwaspAgenticItem(
        id="ASI03",
        key=ASI03,
        name="Identity & Privilege Abuse",
        description="Self-claimed authority, forged permissions, cached credential abuse, and cross-user access.",
        covered_by_existing_modules=["ToolPolicyChecker"],
        dataset_files=["identity_privilege_abuse.yaml"],
        detector_modules=["IdentityGuard"],
        defense_modules=["IdentityGuard", "ToolPolicyChecker"],
        runtime_hooks=["user_input", "tool_call", "tool_output", "memory_read"],
        current_status="medium",
    ),
    "ASI04": OwaspAgenticItem(
        id="ASI04",
        key=ASI04,
        name="Agentic Supply Chain Vulnerabilities",
        description="Poisoned tool descriptors, skills, dependencies, model artifacts, or runtime configuration.",
        covered_by_existing_modules=[],
        dataset_files=["supply_chain.yaml"],
        detector_modules=["SupplyChainGuard"],
        defense_modules=["SupplyChainGuard"],
        runtime_hooks=["supply_chain_load", "memory_read", "tool_call"],
        current_status="medium",
    ),
    "ASI05": OwaspAgenticItem(
        id="ASI05",
        key=ASI05,
        name="Unexpected Code Execution",
        description="Shell execution, generated code execution, CI/CD mutation, SQL mutation, or package install abuse.",
        covered_by_existing_modules=["ToolPolicyChecker"],
        dataset_files=["code_execution.yaml"],
        detector_modules=["CodeExecutionGuard"],
        defense_modules=["CodeExecutionGuard", "ToolPolicyChecker"],
        runtime_hooks=["tool_call", "tool_output", "rag_retrieval"],
        current_status="strong",
    ),
    "ASI06": OwaspAgenticItem(
        id="ASI06",
        key=ASI06,
        name="Memory & Context Poisoning",
        description="Poisoned RAG, memory, metadata, summaries, or context that corrupts future behavior.",
        covered_by_existing_modules=["RAGSanitizer", "MemorySafetyChecker"],
        dataset_files=["rag_poisoning.yaml"],
        detector_modules=["RAGSanitizer", "MemorySafetyChecker"],
        defense_modules=["RAGSanitizer", "MemorySafetyChecker"],
        runtime_hooks=["rag_retrieval", "rag_metadata", "memory_read", "memory_write"],
        current_status="strong",
    ),
    "ASI07": OwaspAgenticItem(
        id="ASI07",
        key=ASI07,
        name="Insecure Inter-Agent Communication",
        description="Spoofed agent messages, role confusion, and untrusted worker outputs in multi-agent systems.",
        covered_by_existing_modules=[],
        dataset_files=["inter_agent_communication.yaml"],
        detector_modules=["InterAgentGuard"],
        defense_modules=["InterAgentGuard"],
        runtime_hooks=["inter_agent_message"],
        current_status="medium",
    ),
    "ASI08": OwaspAgenticItem(
        id="ASI08",
        key=ASI08,
        name="Cascading Failures",
        description="Risk propagation from input, RAG, tool, answer, memory, or repair loops.",
        covered_by_existing_modules=["MemorySafetyChecker", "AnswerVerifier"],
        dataset_files=["cascading_failures.yaml"],
        detector_modules=["CascadeGuard"],
        defense_modules=["CascadeGuard", "MemorySafetyChecker"],
        runtime_hooks=["memory_write", "tool_output", "final_output", "autonomous_loop"],
        current_status="medium",
    ),
    "ASI09": OwaspAgenticItem(
        id="ASI09",
        key=ASI09,
        name="Human-Agent Trust Exploitation",
        description="Risk downplaying, misleading confirmation text, fake authority, and overconfident unsafe guidance.",
        covered_by_existing_modules=["OutputGuard", "ToolPolicyChecker"],
        dataset_files=["human_trust_exploitation.yaml"],
        detector_modules=["HumanTrustGuard"],
        defense_modules=["HumanTrustGuard", "ToolPolicyChecker"],
        runtime_hooks=["final_output", "tool_call"],
        current_status="medium",
    ),
    "ASI10": OwaspAgenticItem(
        id="ASI10",
        key=ASI10,
        name="Rogue Agents",
        description="Goal drift, excessive autonomy, self-modification, and anomalous long-running behavior.",
        covered_by_existing_modules=[],
        dataset_files=["rogue_agent.yaml"],
        detector_modules=["RogueAgentGuard"],
        defense_modules=["RogueAgentGuard"],
        runtime_hooks=["autonomous_loop", "tool_call", "memory_write"],
        current_status="medium",
    ),
}


def normalize_owasp_ids(value: Any, category: str = "") -> List[str]:
    if isinstance(value, str) and value.strip():
        raw_items = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_items = [str(item) for item in value if str(item).strip()]
    else:
        raw_items = []

    normalized = []
    for item in raw_items:
        upper = str(item).strip().upper()
        if upper in OWASP_AGENTIC_TOP10:
            normalized.append(upper)
            continue
        for asi_id, entry in OWASP_AGENTIC_TOP10.items():
            if item == entry.key:
                normalized.append(asi_id)
                break

    if normalized:
        return sorted(set(normalized))
    return category_to_owasp_ids(category)


def category_to_owasp_ids(category: str) -> List[str]:
    mapping = {
        "prompt_injection": ["ASI01"],
        "jailbreak": ["ASI01"],
        "tool_misuse": ["ASI02"],
        "identity_privilege_abuse": ["ASI03"],
        "supply_chain": ["ASI04"],
        "code_execution": ["ASI05"],
        "rag_poisoning": ["ASI06"],
        "memory_context_poisoning": ["ASI06"],
        "inter_agent_communication": ["ASI07"],
        "cascading_failures": ["ASI08"],
        "human_trust_exploitation": ["ASI09"],
        "human_trust": ["ASI09"],
        "rogue_agent": ["ASI10"],
    }
    return mapping.get(str(category or "").strip(), [])


def coverage_matrix_template() -> Dict[str, Dict[str, Any]]:
    return {asi_id: entry.to_dict() for asi_id, entry in OWASP_AGENTIC_TOP10.items()}
