"""EVO Orchestrator Adapter."""
from __future__ import annotations
import logging
import subprocess
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

class OrchestratorAdapter:
    """Adapter for EVO orchestrator (hermes chat -q based agents)."""
    
    def __init__(self, hermes_path: str = "hermes", git_bash_path: str = "") -> None:
        self._hermes_path = hermes_path
        self._git_bash_path = git_bash_path
        self._agents: dict[str, dict] = {}
    
    def spawn_agent(self, agent_id: str, role: str = "worker", model: str = "",
                    prompt: str = "") -> bool:
        self._agents[agent_id] = {"role": role, "model": model, "prompt": prompt,
                                  "spawned_at": time.time(), "status": "active"}
        logger.info(f"Agent spawned: {agent_id} ({role})")
        return True
    
    def send_task(self, agent_id: str, task: str) -> str | None:
        agent = self._agents.get(agent_id)
        if not agent:
            return None
        try:
            cmd = [self._hermes_path, "chat", "-q", task]
            if agent.get("model"):
                cmd.extend(["--model", agent["model"]])
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return result.stdout[:2000] if result.returncode == 0 else None
        except Exception as e:
            logger.warning(f"Agent task error: {e}")
            return None
    
    def list_agents(self) -> list[dict]:
        return [{"id": aid, **info} for aid, info in self._agents.items()]
    
    def terminate_agent(self, agent_id: str) -> bool:
        if agent_id in self._agents:
            self._agents[agent_id]["status"] = "terminated"
            return True
        return False
