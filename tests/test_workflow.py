"""Unit tests for WorkflowManager."""

from __future__ import annotations

import pytest

from comfyclaw.workflow import WorkflowManager


class TestAddNode:
    def test_returns_string_id(self, wm: WorkflowManager) -> None:
        nid = wm.add_node("VAEDecode")
        assert isinstance(nid, str)

    def test_increments_ids_sequentially(self, wm: WorkflowManager) -> None:
        existing_max = max(int(k) for k in wm.workflow)
        id1 = wm.add_node("A")
        id2 = wm.add_node("B")
        id3 = wm.add_node("C")
        assert int(id1) == existing_max + 1
        assert int(id2) == existing_max + 2
        assert int(id3) == existing_max + 3

    def test_node_appears_in_workflow(self, wm: WorkflowManager) -> None:
        nid = wm.add_node("SaveImage", nickname="Output", filename_prefix="test")
        assert nid in wm.workflow
        node = wm.workflow[nid]
        assert node["class_type"] == "SaveImage"
        assert node["_meta"]["title"] == "Output"
        assert node["inputs"]["filename_prefix"] == "test"

    def test_nickname_defaults_to_class_type(self, wm: WorkflowManager) -> None:
        nid = wm.add_node("VAELoader")
        assert wm.workflow[nid]["_meta"]["title"] == "VAELoader"


class TestConnect:
    def test_wires_input(self, wm: WorkflowManager) -> None:
        nid = wm.add_node("VAEDecode", samples_placeholder="x")
        wm.connect("3", 0, nid, "samples")
        assert wm.workflow[nid]["inputs"]["samples"] == ["3", 0]

    def test_overwrites_existing_link(self, wm: WorkflowManager) -> None:
        # node "3" already has model ← ["1", 0]; rewire to a new source
        new_src = wm.add_node("UNETLoader")
        wm.connect(new_src, 0, "3", "model")
        assert wm.workflow["3"]["inputs"]["model"] == [new_src, 0]

    def test_raises_on_missing_dst(self, wm: WorkflowManager) -> None:
        with pytest.raises(KeyError, match="999"):
            wm.connect("1", 0, "999", "model")


class TestSetParam:
    def test_updates_scalar(self, wm: WorkflowManager) -> None:
        wm.set_param("3", "steps", 30)
        assert wm.workflow["3"]["inputs"]["steps"] == 30

    def test_raises_on_missing_node(self, wm: WorkflowManager) -> None:
        with pytest.raises(KeyError):
            wm.set_param("999", "steps", 10)

    def test_adds_new_param(self, wm: WorkflowManager) -> None:
        wm.set_param("3", "new_key", "hello")
        assert wm.workflow["3"]["inputs"]["new_key"] == "hello"


class TestDeleteNode:
    def test_removes_node(self, wm: WorkflowManager) -> None:
        assert "1" in wm.workflow
        wm.delete_node("1")
        assert "1" not in wm.workflow

    def test_removes_dangling_links(self, wm: WorkflowManager) -> None:
        # "2" has clip ← ["1", 1]; "3" has model ← ["1", 0] and positive ← ["2", 0]
        wm.delete_node("1")
        assert "clip" not in wm.workflow["2"]["inputs"]
        assert "model" not in wm.workflow["3"]["inputs"]
        # "positive" link to "2" (still exists) should remain
        assert "positive" in wm.workflow["3"]["inputs"]

    def test_raises_on_missing_node(self, wm: WorkflowManager) -> None:
        with pytest.raises(KeyError):
            wm.delete_node("999")


class TestGetNodesByClass:
    def test_finds_matching(self, wm: WorkflowManager) -> None:
        ids = wm.get_nodes_by_class("CLIPTextEncode")
        assert ids == ["2"]

    def test_finds_multiple(self, wm: WorkflowManager) -> None:
        wm.add_node("CLIPTextEncode", clip=["1", 1], text="negative")
        ids = wm.get_nodes_by_class("CLIPTextEncode")
        assert len(ids) == 2

    def test_empty_on_no_match(self, wm: WorkflowManager) -> None:
        assert wm.get_nodes_by_class("NoSuchNode") == []


class TestGetNodeByTitle:
    def test_finds_existing(self, wm: WorkflowManager) -> None:
        assert wm.get_node_by_title("KSampler") == "3"

    def test_returns_none_for_missing(self, wm: WorkflowManager) -> None:
        assert wm.get_node_by_title("NonExistent") is None


class TestValidate:
    def test_valid_workflow_no_errors(self, wm: WorkflowManager) -> None:
        assert wm.validate() == []

    def test_detects_dangling_link(self, wm: WorkflowManager) -> None:
        wm.workflow["3"]["inputs"]["model"] = ["999", 0]  # points to non-existent node
        errors = wm.validate()
        assert len(errors) == 1
        assert "999" in errors[0]
        assert "3" in errors[0]

    def test_no_false_positives_for_scalars(self, wm: WorkflowManager) -> None:
        wm.set_param("3", "cfg", 7.5)
        assert wm.validate() == []


class TestClone:
    def test_clone_is_independent(self, wm: WorkflowManager) -> None:
        clone = wm.clone()
        clone.set_param("3", "steps", 99)
        assert wm.workflow["3"]["inputs"]["steps"] == 20  # original unchanged

    def test_clone_has_same_content(self, wm: WorkflowManager) -> None:
        clone = wm.clone()
        assert clone.workflow == wm.workflow

    def test_clone_counter_continues(self, wm: WorkflowManager) -> None:
        clone = wm.clone()
        nid = clone.add_node("VAEDecode")
        assert int(nid) > 3  # should be 4 (existing max + 1)


class TestApplyImageModel:
    def test_updates_checkpoint_loader(self) -> None:
        wm = WorkflowManager(
            {
                "1": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {"ckpt_name": "old_model.safetensors"},
                },
            }
        )
        updated = wm.apply_image_model("new_model.safetensors")
        assert len(updated) == 1
        assert updated[0] == ("1", "ckpt_name")
        assert wm.workflow["1"]["inputs"]["ckpt_name"] == "new_model.safetensors"

    def test_updates_unet_loader(self) -> None:
        wm = WorkflowManager(
            {
                "2": {
                    "class_type": "UNETLoader",
                    "inputs": {"unet_name": "old.pt", "weight_dtype": "default"},
                },
            }
        )
        updated = wm.apply_image_model("Qwen/Qwen-Image-2512")
        assert updated[0] == ("2", "unet_name")
        assert wm.workflow["2"]["inputs"]["unet_name"] == "Qwen/Qwen-Image-2512"
        # weight_dtype must be untouched
        assert wm.workflow["2"]["inputs"]["weight_dtype"] == "default"

    def test_updates_multiple_loaders(self) -> None:
        wm = WorkflowManager(
            {
                "1": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {"ckpt_name": "a.safetensors"},
                },
                "2": {"class_type": "UNETLoader", "inputs": {"unet_name": "b.pt"}},
            }
        )
        updated = wm.apply_image_model("target_model.safetensors")
        assert len(updated) == 2
        assert wm.workflow["1"]["inputs"]["ckpt_name"] == "target_model.safetensors"
        assert wm.workflow["2"]["inputs"]["unet_name"] == "target_model.safetensors"

    def test_ignores_lora_loader(self) -> None:
        wm = WorkflowManager(
            {
                "1": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {"ckpt_name": "base.safetensors"},
                },
                "2": {"class_type": "LoraLoader", "inputs": {"lora_name": "style.safetensors"}},
            }
        )
        updated = wm.apply_image_model("new_base.safetensors")
        assert len(updated) == 1  # only the checkpoint, not the LoRA
        assert wm.workflow["2"]["inputs"]["lora_name"] == "style.safetensors"

    def test_returns_empty_when_no_loaders(self) -> None:
        wm = WorkflowManager(
            {
                "1": {"class_type": "KSampler", "inputs": {"steps": 20}},
            }
        )
        updated = wm.apply_image_model("any_model.safetensors")
        assert updated == []

    def test_creates_inputs_dict_if_missing(self) -> None:
        wm = WorkflowManager(
            {
                "1": {"class_type": "CheckpointLoaderSimple"},
            }
        )
        wm.apply_image_model("fresh_model.safetensors")
        assert wm.workflow["1"]["inputs"]["ckpt_name"] == "fresh_model.safetensors"


class TestSerialization:
    def test_to_dict_is_deep_copy(self, wm: WorkflowManager) -> None:
        d = wm.to_dict()
        d["3"]["inputs"]["steps"] = 99
        assert wm.workflow["3"]["inputs"]["steps"] == 20

    def test_from_json_roundtrip(self, wm: WorkflowManager) -> None:
        json_str = wm.to_json()
        restored = WorkflowManager.from_json(json_str)
        assert restored.workflow == wm.workflow

    def test_len(self, wm: WorkflowManager) -> None:
        assert len(wm) == 3
