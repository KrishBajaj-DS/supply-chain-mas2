"""
Peer-to-Peer (P2P) Messaging Architecture Implementation.

Requirement 1:
"Different Messaging Architecture needs to be tried but Blackboard(Shared State)
is mandatory. Example Can Blackboard architecture Vs Peer to Peer?. Blackboard/context shared state is MUST."

In Peer-to-Peer (P2P) Messaging:
1. No Central Shared State: There is no shared blackboard or central memory.
2. Direct Point-to-Point Messaging: Agents send messages directly to specific peer inboxes.
3. Information Handoff ("Telephone Game"): Downstream peers only receive what their
   immediate upstream peer forwards to them, which can lead to context loss or attenuation
   compared to Blackboard architecture.
"""

from typing import Dict, List, Any, Optional
import time
from datetime import datetime


class P2PMessage:
    """Represents a direct point-to-point message between two agent peers."""
    def __init__(self, sender: str, recipient: str, payload: Any, metadata: Optional[Dict] = None):
        self.sender = sender
        self.recipient = recipient
        self.payload = payload
        self.metadata = metadata or {}
        self.timestamp = time.time()
        self.iso_time = datetime.fromtimestamp(self.timestamp).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "iso_time": self.iso_time,
        }


class P2PAgentNode:
    """An agent node participating in a peer-to-peer network with private inbox/outbox."""
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.inbox: List[P2PMessage] = []
        self.outbox: List[P2PMessage] = []

    def receive_message(self, message: P2PMessage) -> None:
        self.inbox.append(message)

    def send_message(self, recipient_node: 'P2PAgentNode', payload: Any) -> P2PMessage:
        msg = P2PMessage(sender=self.name, recipient=recipient_node.name, payload=payload)
        self.outbox.append(msg)
        recipient_node.receive_message(msg)
        return msg

    def get_latest_message(self) -> Optional[P2PMessage]:
        return self.inbox[-1] if self.inbox else None


class P2PNetwork:
    """
    Manages peer-to-peer direct routing between agents.
    Tracks point-to-point message traffic without maintaining any central shared state.
    """
    def __init__(self):
        self.nodes: Dict[str, P2PAgentNode] = {}
        self.message_traffic: List[P2PMessage] = []

    def register_node(self, node: P2PAgentNode) -> None:
        self.nodes[node.name] = node

    def route_message(self, sender_name: str, recipient_name: str, payload: Any) -> P2PMessage:
        sender = self.nodes.get(sender_name)
        recipient = self.nodes.get(recipient_name)
        if not sender or not recipient:
            raise ValueError(f"Both {sender_name} and {recipient_name} must be registered in P2P network")
        msg = sender.send_message(recipient, payload)
        self.message_traffic.append(msg)
        return msg

    def get_traffic_log(self) -> List[Dict[str, Any]]:
        return [msg.to_dict() for msg in self.message_traffic]
