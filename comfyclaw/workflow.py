"""
WorkflowManager — in-memory manipulation of ComfyUI API-format workflows.

The API format is a flat dict keyed by string node IDs:

    {
      "1": {"class_type": "CheckpointLoaderSimple", "inputs": {...}, "_meta": {...}},
      "2": {"class_type": "CLIPTextEncode",          "inputs": {"clip": ["1", 1], "text": "..."}, ...},
    }

Link references are ``[src_node_id_str, output_slot_index]`` tuples stored
directly in the destination node's ``inputs`` dict.
"""

from __future__ import annotations

import copy
import json
from typing import Any


class WorkflowValidationError(Exception):
    """Raised when a workflow contains structural problems."""


class WorkflowManager:
    """
    Mutable in-memory representation of a ComfyUI API-format workflow.

    Parameters
    ----------
    workflow : Initial workflow dict (copied on construction).
    """

    def __init__(self, workflow: dict | None = None) -> None:
        self.workflow: dict[str, dict] = copy.deepcopy(workflow or {})
        self._sync_counter()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_counter(self) -> None:
        ids = [int(k) for k in self.workflow if k.isdigit()]
        self._next_id = max(ids, default=0)

    def _new_id(self) -> str:
        self._next_id += 1
        return str(self._next_id)

    @staticmethod
    def _is_link(value: Any) -> bool:
        return (
            isinstance(value, list)
            and len(value) == 2
            and isinstance(value[0], str | int)
            and isinstance(value[1], int)
        )

    # ------------------------------------------------------------------
    # Graph mutations
    # ------------------------------------------------------------------

    def add_node(
        self,
        class_type: str,
        nickname: str | None = None,
        **inputs: Any,
    ) -> str:
        """
        Append a new node and return its ID.

        Parameters
        ----------
        class_type : ComfyUI class name (e.g. ``"KSampler"``).
        nickname   : Human-readable title stored in ``_meta.title``.
        **inputs   : Initial input values (scalars or link tuples).
        """
        node_id = self._new_id()
        self.workflow[node_id] = {
            "class_type": class_type,
            "_meta": {"title": nickname or class_type},
            "inputs": dict(inputs),
        }
        return node_id

    def connect(
        self,
        src_node_id: str,
        src_output_index: int,
        dst_node_id: str,
        dst_input_name: str,
    ) -> None:
        """Wire ``src_node_id[src_output_index]`` → ``dst_node_id.dst_input_name``."""
        if dst_node_id not in self.workflow:
            raise KeyError(f"Destination node {dst_node_id!r} not found in workflow")
        self.workflow[dst_node_id]["inputs"][dst_input_name] = [
            str(src_node_id),
            src_output_index,
        ]

    def set_param(self, node_id: str, param_name: str, value: Any) -> None:
        """Set a scalar input on a node."""
        if node_id not in self.workflow:
            raise KeyError(f"Node {node_id!r} not found in workflow")
        self.workflow[node_id]["inputs"][param_name] = value

    def delete_node(self, node_id: str) -> None:
        """
        Remove a node and clean up all dangling link references in other nodes.
        """
        if node_id not in self.workflow:
            raise KeyError(f"Node {node_id!r} not found in workflow")
        del self.workflow[node_id]
        for node in self.workflow.values():
            stale = [
                k
                for k, v in node.get("inputs", {}).items()
                if self._is_link(v) and str(v[0]) == str(node_id)
            ]
            for k in stale:
                del node["inputs"][k]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Model overrides
    # ------------------------------------------------------------------

    #: Ordered list of (class_type, input_param) pairs for every ComfyUI
    #: checkpoint / UNET loader node that holds a model filename.
    _LOADER_PARAMS: list[tuple[str, str]] = [
        ("CheckpointLoaderSimple", "ckpt_name"),
        ("CheckpointLoader", "ckpt_name"),
        ("UNETLoader", "unet_name"),
        ("unCLIPCheckpointLoader", "ckpt_name"),
        ("Hy3DCheckpointLoader", "ckpt_name"),
        ("ImageOnlyCheckpointLoader", "ckpt_name"),
    ]

    def apply_image_model(self, model_name: str) -> list[tuple[str, str]]:
        """
        Pin the image-generation model across all matching loader nodes.

        Scans the workflow for known checkpoint / UNET loader classes and sets
        the relevant parameter to *model_name* on every match found.

        Parameters
        ----------
        model_name :
            The model filename or HuggingFace-style path, e.g.
            ``"Qwen/Qwen-Image-2512"`` or ``"realisticVision_v51.safetensors"``.

        Returns
        -------
        List of ``(node_id, param_name)`` tuples that were updated.
        Empty list if no loader nodes were found (the workflow may load models
        in a custom way — inspect and pin manually in that case).
        """
        updated: list[tuple[str, str]] = []
        known_params = dict(self._LOADER_PARAMS)
        for nid, node in self.workflow.items():
            ct = node.get("class_type", "")
            if ct in known_params:
                param = known_params[ct]
                node.setdefault("inputs", {})[param] = model_name
                updated.append((nid, param))
        return updated

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_nodes_by_class(self, class_type: str) -> list[str]:
        """Return node IDs whose ``class_type`` matches *class_type*."""
        return [nid for nid, node in self.workflow.items() if node.get("class_type") == class_type]

    def get_node_by_title(self, title: str) -> str | None:
        """Return the first node ID whose ``_meta.title`` matches, or None."""
        for nid, node in self.workflow.items():
            if node.get("_meta", {}).get("title") == title:
                return nid
        return None

    def validate(self) -> list[str]:
        """
        Check referential integrity.

        Returns a (possibly empty) list of human-readable error strings
        describing dangling or invalid links.
        """
        errors: list[str] = []
        for nid, node in self.workflow.items():
            for inp_name, inp_val in node.get("inputs", {}).items():
                if not self._is_link(inp_val):
                    continue
                src_id = str(inp_val[0])
                if src_id not in self.workflow:
                    errors.append(
                        f"Node {nid!r} input {inp_name!r} references missing node {src_id!r}"
                    )
        return errors

    # ------------------------------------------------------------------
    # Serialisation / cloning
    # ------------------------------------------------------------------

    def clone(self) -> WorkflowManager:
        """Return a deep-copy of this manager."""
        return WorkflowManager(copy.deepcopy(self.workflow))

    def to_dict(self) -> dict:
        """Return a deep copy of the workflow dict (safe to mutate)."""
        return copy.deepcopy(self.workflow)

    def to_json(self, **kwargs: Any) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.workflow, **kwargs)

    def __len__(self) -> int:
        return len(self.workflow)

    def __repr__(self) -> str:
        classes = {n.get("class_type", "?") for n in self.workflow.values()}
        return f"<WorkflowManager nodes={len(self)} classes={sorted(classes)}>"

    # ------------------------------------------------------------------
    # Class methods / static helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, json_str: str) -> WorkflowManager:
        return cls(json.loads(json_str))

    @classmethod
    def from_file(cls, path: str) -> WorkflowManager:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        # Handle legacy "prompt"-keyed save format
        if isinstance(data, dict) and "prompt" in data and isinstance(data["prompt"], dict):
            data = data["prompt"]
        return cls(data)

    @staticmethod
    def summarize(workflow: dict) -> str:
        """
        Return a compact, human-readable summary of a workflow dict.
        Useful for injecting into an LLM prompt.
        """
        if not workflow:
            return "(empty workflow)"
        lines = ["Workflow nodes:"]
        for nid in sorted(workflow.keys(), key=lambda x: int(x) if x.isdigit() else 0):
            node = workflow[nid]
            title = node.get("_meta", {}).get("title") or node.get("class_type", "?")
            inputs_repr: list[str] = []
            for k, v in node.get("inputs", {}).items():
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                    inputs_repr.append(f"{k}=→node{v[0]}[{v[1]}]")
                else:
                    v_s = repr(v) if isinstance(v, str) else json.dumps(v)
                    inputs_repr.append(f"{k}={v_s}")
            lines.append(
                f"  [{nid}] {title} ({node.get('class_type', '?')})"
                + (f"  {', '.join(inputs_repr)}" if inputs_repr else "")
            )
        return "\n".join(lines)
