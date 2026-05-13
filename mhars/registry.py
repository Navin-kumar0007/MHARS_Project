import os
import json
import time
from typing import Dict, Any

class AgentRegistry:
    """
    Lightweight, file-based registry for multi-agent coordination.
    Allows multiple edge nodes running MHARS to discover each other
    and share basic state without requiring Redis or external databases.
    """
    def __init__(self, registry_file: str = "logs/registry.json"):
        self.registry_file = os.path.join(os.path.dirname(__file__), '..', registry_file)
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        
        if not os.path.exists(self.registry_file):
            self._save({})

    def _load(self) -> Dict[str, Any]:
        try:
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save(self, data: Dict[str, Any]):
        tmp_file = self.registry_file + ".tmp"
        with open(tmp_file, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_file, self.registry_file)

    def register_node(self, node_id: str, machine_type: str, status: str = "active"):
        """Register or update a node's heartbeat in the registry."""
        registry = self._load()
        registry[node_id] = {
            "machine_type": machine_type,
            "status": status,
            "last_heartbeat": time.time(),
            "last_updated": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
        }
        self._save(registry)

    def get_active_nodes(self, timeout_seconds: int = 300) -> Dict[str, Any]:
        """Return all nodes that have pinged within the timeout window."""
        registry = self._load()
        now = time.time()
        active = {}
        for node_id, data in registry.items():
            if (now - data.get("last_heartbeat", 0)) < timeout_seconds:
                active[node_id] = data
        return active
    def list_all_nodes(self) -> Dict[str, Any]:
        """Return all nodes currently in the registry, regardless of timeout."""
        return self._load()
