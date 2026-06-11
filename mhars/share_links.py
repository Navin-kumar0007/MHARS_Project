import os
import json
import uuid
import time
from typing import Dict, Any, Optional

SHARE_LINKS_FILE = os.path.join(os.path.dirname(__file__), "share_links.json")

class ShareLinkManager:
    """
    Manages shareable, public status links for remote monitoring.
    """
    
    def __init__(self):
        self._ensure_file()
        
    def _ensure_file(self):
        if not os.path.exists(SHARE_LINKS_FILE):
            with open(SHARE_LINKS_FILE, "w") as f:
                json.dump({}, f)
                
    def _load(self) -> Dict[str, Any]:
        try:
            with open(SHARE_LINKS_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
            
    def _save(self, data: Dict[str, Any]):
        with open(SHARE_LINKS_FILE, "w") as f:
            json.dump(data, f, indent=4)
            
    def create_link(self, creator: str, label: str, expires_in_hours: int = 24) -> str:
        """Creates a new share link token."""
        links = self._load()
        
        # Rate limit check (max 10 active per user)
        active_user_links = [
            t for t, d in links.items() 
            if d.get("creator") == creator and d.get("expires_at", 0) > time.time()
        ]
        
        if len(active_user_links) >= 10:
            raise ValueError("Maximum of 10 active share links allowed per user.")
            
        token = str(uuid.uuid4())
        
        links[token] = {
            "creator": creator,
            "label": label,
            "created_at": time.time(),
            "expires_at": time.time() + (expires_in_hours * 3600),
            "access_count": 0
        }
        
        self._save(links)
        return token
        
    def validate_and_record_access(self, token: str) -> bool:
        """Checks if a token is valid, and increments access count."""
        links = self._load()
        
        if token not in links:
            return False
            
        link_data = links[token]
        
        if time.time() > link_data.get("expires_at", 0):
            # Cleanup expired
            del links[token]
            self._save(links)
            return False
            
        # Record access
        link_data["access_count"] = link_data.get("access_count", 0) + 1
        link_data["last_accessed"] = time.time()
        self._save(links)
        
        return True
        
    def list_links(self) -> list:
        """Returns all active links."""
        links = self._load()
        now = time.time()
        
        active_links = []
        cleanup_needed = False
        
        for token, data in list(links.items()):
            if now > data.get("expires_at", 0):
                del links[token]
                cleanup_needed = True
            else:
                active_links.append({"token": token, **data})
                
        if cleanup_needed:
            self._save(links)
            
        return active_links
        
    def revoke_link(self, token: str) -> bool:
        """Revokes a specific token."""
        links = self._load()
        if token in links:
            del links[token]
            self._save(links)
            return True
        return False
