import pytest
from pydantic import ValidationError

from conftest import CONFIG
from mac_sop_comparator.config import ArchitectureParams, SOURCED_PARAMS, parse_config

BASE = dict(weight_bits=32, e_mac=1.0, e_ac=0.2, e_upd=0.3, e_byte_von_neumann=53.0, e_byte_neuromorphic=1.3)


def test_defaults_follow_d23():
    p = ArchitectureParams(**BASE)
    assert (p.b_a, p.b_s, p.b_e, p.b_o, p.inference_batch) == (4, 4, 4, 4, 1)
    assert p.sources == {}


@pytest.mark.parametrize("bits,expected", [(32, 4.0), (16, 2.0), (8, 1.0), (4, 0.5)])
def test_weight_bytes_derive_from_bits(bits, expected):
    assert ArchitectureParams(**{**BASE, "weight_bits": bits}).b_w == expected


@pytest.mark.parametrize("field", ["e_byte_von_neumann", "e_byte_neuromorphic"])
def test_byte_energies_are_required(field):
    values = {k: v for k, v in BASE.items() if k != field}
    with pytest.raises(ValidationError, match=field):
        ArchitectureParams(**values)


@pytest.mark.parametrize("field,value", [("e_byte_von_neumann", -1.0), ("b_s", 0), ("inference_batch", 0)])
def test_invalid_values_are_rejected(field, value):
    with pytest.raises(ValidationError, match=field):
        ArchitectureParams(**{**BASE, field: value})


def test_missing_sources_lists_every_parameter_without_one():
    assert ArchitectureParams(**BASE).missing_sources() == list(SOURCED_PARAMS)
    full = {name: "ref" for name in SOURCED_PARAMS}
    assert ArchitectureParams(**BASE, sources=full).missing_sources() == []
    partial = ArchitectureParams(**BASE, sources={"e_mac": "Horowitz", "e_ac": ""})
    assert "e_mac" not in partial.missing_sources() and "e_ac" in partial.missing_sources()


def test_sources_with_unknown_parameter_are_rejected():
    with pytest.raises(ValidationError, match="e_macs"):
        ArchitectureParams(**BASE, sources={"e_macs": "typo"})


def test_old_memory_scenarios_key_is_now_an_error():
    with pytest.raises(ValidationError, match="memory_scenarios"):
        ArchitectureParams(**BASE, memory_scenarios=["von_neumann"])


def test_shipped_configs_cite_every_source():
    from pathlib import Path

    from mac_sop_comparator.config import load_config

    for name in ("tox21_sr_are.yaml", "tox21_sr_are_2048.yaml"):
        cfg = load_config(Path(__file__).parent.parent / "configs" / name)
        assert cfg.architecture.missing_sources() == [], name
        assert "Davies" in cfg.architecture.sources["e_byte_neuromorphic"]
        assert "Horowitz" in cfg.architecture.sources["e_byte_von_neumann"]


def test_config_without_sources_parses():
    assert parse_config(CONFIG).architecture.sources == {}
