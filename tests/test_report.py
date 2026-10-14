import json

import pytest

from mac_sop_comparator.profiling import OpReport

ANN = dict(architecture="ann", macs=139_392, biases=194)
SNN = dict(architecture="snn", sops=1234.5, state_updates=2540.0, spikes_per_layer=[26.0, 40.5, 12.25, 3.0], biases=194)


def test_ann_report():
    r = OpReport(**ANN)
    assert (r.macs, r.sops, r.state_updates, r.spikes_per_layer, r.bytes_accessed) == (139_392, None, None, None, None)


def test_snn_report():
    r = OpReport(**SNN)
    assert (r.sops, r.state_updates, r.macs) == (1234.5, 2540.0, None)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(architecture="cnn"), "architecture"),
        (dict(architecture="ann"), "needs"),  # no macs
        (dict(architecture="ann", macs=1, sops=2.0), "cannot have"),
        (dict(architecture="snn", sops=1.0), "needs"),  # no state_updates / spikes
        (dict(architecture="snn", sops=1.0, state_updates=1.0, spikes_per_layer=[1.0], macs=5), "cannot have"),
        (dict(architecture="ann", macs=-1), "macs"),
        (dict(architecture="snn", sops=1.0, state_updates=-2.0, spikes_per_layer=[1.0]), "state_updates"),
        (dict(architecture="snn", sops=1.0, state_updates=1.0, spikes_per_layer=[-1.0]), "spikes_per_layer"),
        (dict(architecture="ann", macs=1, bytes_accessed={"gpu": 1.0}), "bytes_accessed"),
    ],
)
def test_invalid_reports(kwargs, message):
    with pytest.raises(ValueError, match=message):
        OpReport(**kwargs)


def test_json_schema_matches_the_design():
    d = OpReport(**SNN, bytes_accessed={"von_neumann": 10.0, "neuromorphic": 4.0}).to_dict()
    assert list(d) == ["architecture", "per_inference"] and d["architecture"] == "snn"
    assert set(d["per_inference"]) == {"macs", "sops", "state_updates", "spikes_per_layer", "biases", "bytes_accessed"}
    assert d["per_inference"]["bytes_accessed"] == {"von_neumann": 10.0, "neuromorphic": 4.0}
    json.dumps(d)  # serializable


@pytest.mark.parametrize("kwargs", [ANN, SNN, {**ANN, "bytes_accessed": {"von_neumann": 1.0}}])
def test_dict_and_file_roundtrip(kwargs, tmp_path):
    r = OpReport(**kwargs)
    assert OpReport.from_dict(r.to_dict()) == r
    path = r.save(tmp_path / "sub" / "opreport_x.json")
    assert OpReport.load(path) == r
    assert json.loads(path.read_text())["per_inference"]["bytes_accessed"] == kwargs.get("bytes_accessed")


def test_report_is_immutable():
    with pytest.raises(Exception):
        OpReport(**ANN).macs = 1
