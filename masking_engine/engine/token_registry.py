"""
token_registry.py — Idempotent Session Token Registry
R26-CS-012: Context-Aware Masking + Instruction Engine

Implements the session-scoped idempotent token map (Taxonomy Section 10).

Rules:
  1. Same value + same session  → same token  (idempotent)
  2. Different value, same type → incremented token (<API_KEY_2>)
  3. Session ends               → registry destroyed
  4. Registry stored locally only — never transmitted to external AI
"""

import uuid
from datetime import datetime
from typing import Optional, Dict


class TokenRegistry:
    """
    In-memory session-scoped token registry.

    Maintains two indexes:
      - value_to_token : original_value → token_label
      - token_to_meta  : token_label → {original, entity_type, first_seen, last_seen}
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id      = session_id or f"sess_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"
        self._value_to_token : Dict[str, str] = {}
        self._token_to_meta  : Dict[str, dict] = {}
        self._type_counters  : Dict[str, int]  = {}  # entity_type → current count
        self._prompt_counter : int             = 0

    def next_prompt(self):
        """Call at the start of each new prompt in a session."""
        self._prompt_counter += 1

    def get_or_create_token(self, value: str, entity_type: str) -> str:
        """
        Idempotent: if value already has a token, return it.
        Otherwise create a new <TYPE_N> token and register it.
        """
        if value in self._value_to_token:
            token = self._value_to_token[value]
            # Update last_seen
            self._token_to_meta[token]["last_seen_prompt"] = self._prompt_counter
            return token

        # New value — increment counter for this type
        count = self._type_counters.get(entity_type, 0) + 1
        self._type_counters[entity_type] = count

        # Build token label: <ENTITY_TYPE_N>
        # Shorten type name for readability
        short = entity_type.replace("_", "")[:10].upper()
        token = f"<{short}_{count}>"

        self._value_to_token[value] = token
        self._token_to_meta[token]  = {
            "original"          : value,
            "entity_type"       : entity_type,
            "first_seen_prompt" : self._prompt_counter,
            "last_seen_prompt"  : self._prompt_counter,
        }

        return token

    def get_original(self, token: str) -> Optional[str]:
        """Reverse-lookup: token → original value (for local use only)."""
        meta = self._token_to_meta.get(token)
        return meta["original"] if meta else None

    def export(self) -> dict:
        """Export the full session token map (for logging/audit — local only)."""
        return {
            "session_id" : self.session_id,
            "token_map"  : self._token_to_meta,
        }

    def destroy(self):
        """
        Destroy session — clear all maps.
        Called when user session ends. Tokens do not persist across sessions.
        """
        self._value_to_token.clear()
        self._token_to_meta.clear()
        self._type_counters.clear()
        self._prompt_counter = 0

    def summary(self) -> str:
        lines = [
            f"Session    : {self.session_id}",
            f"Tokens     : {len(self._token_to_meta)}",
            f"Prompts    : {self._prompt_counter}",
        ]
        for token, meta in self._token_to_meta.items():
            lines.append(
                f"  {token:20s} → [{meta['entity_type']}] "
                f"first_seen=P{meta['first_seen_prompt']} "
                f"last_seen=P{meta['last_seen_prompt']}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────
# QUICK SELF-TEST — idempotency demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    reg = TokenRegistry()

    key1 = "sk-abcdefghij1234567890abcdefghij12"
    key2 = "sk-zyxwvutsrq9876543210zyxwvutsrq98"

    reg.next_prompt()
    t1a = reg.get_or_create_token(key1, "API_KEY_GENERIC")
    t2a = reg.get_or_create_token(key2, "API_KEY_GENERIC")

    reg.next_prompt()
    t1b = reg.get_or_create_token(key1, "API_KEY_GENERIC")  # same key, prompt 2

    print(f"Key1 first appearance  : {t1a}")
    print(f"Key2 first appearance  : {t2a}")
    print(f"Key1 second appearance : {t1b}  ← same token (idempotent)")
    print(f"Tokens are same: {t1a == t1b}")
    print()
    print(reg.summary())
