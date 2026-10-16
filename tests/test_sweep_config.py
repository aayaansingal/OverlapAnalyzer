import pytest

from overlapanalyzer.sweep.config import ConfigError, parse_sweep_config


def minimal_config(**overrides):
    config = {
        "sweep_name": "test",
        "systems": [{"molecule": "h2", "geometries": [0.7, 1.0]}],
        "analyses": [{"name": "ground", "modes": ["exact"], "poly_degrees": [2, 4]}],
    }
    config.update(overrides)
    return config


def test_expansion_is_cartesian_product():
    config = parse_sweep_config(minimal_config(systems=[
        {"molecule": "h2", "geometries": [0.7, 1.0, 1.3], "basis": ["sto-3g", "6-31g"], "spin": ["s0", "t1"]},
    ]))
    assert len(config.systems) == 3 * 2 * 2
    assert config.experiment_count == 12


def test_ids_are_stable_and_track_physics_only():
    first = parse_sweep_config(minimal_config())
    second = parse_sweep_config(minimal_config())
    assert [s.system_id for s in first.systems] == [s.system_id for s in second.systems]

    other_basis = parse_sweep_config(minimal_config(systems=[{"molecule": "h2", "geometries": [0.7], "basis": "6-31g"}]))
    assert other_basis.systems[0].system_id != first.systems[0].system_id

    # Display-only fields do not change identities
    plots_on = parse_sweep_config(minimal_config(analyses=[{"name": "ground", "modes": ["exact"], "poly_degrees": [2, 4], "save_plots": True}]))
    assert plots_on.analyses[0].analysis_id == first.analyses[0].analysis_id


def test_geometry_ranges():
    linspace = parse_sweep_config(minimal_config(systems=[{"molecule": "h2", "geometries": {"start": 0.5, "stop": 1.5, "num": 5}}]))
    assert [s.geometry for s in linspace.systems] == [0.5, 0.75, 1.0, 1.25, 1.5]
    stepped = parse_sweep_config(minimal_config(systems=[{"molecule": "h2", "geometries": {"start": 0.5, "stop": 1.0, "step": 0.1}}]))
    assert [s.geometry for s in stepped.systems] == [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def test_defaults_apply_and_duplicates_collapse():
    config = parse_sweep_config(minimal_config(
        defaults={"basis": "6-31g"},
        systems=[{"molecule": "h2", "geometries": [0.7]}, {"molecule": "h2", "geometries": [0.7]}],
    ))
    assert len(config.systems) == 1
    assert config.systems[0].basis == "6-31g"


@pytest.mark.parametrize("overrides, message", [
    ({"systms": []}, "unknown key"),
    ({"systems": [{"molecule": "benzene", "geometries": [1.0]}]}, "unknown molecule"),
    ({"systems": [{"molecule": "h_chain", "geometries": [1.0]}]}, "n_atoms"),
    ({"systems": [{"molecule": "h2", "geometries": [1.0], "spin": "q5"}]}, "spin"),
    ({"systems": [{"molecule": "h2"}]}, "needs 'geometries'"),
    ({"systems": [{"molecule": "h2", "geometries": [1.0], "active_space": {"active_electrons": 5, "active_spatial_orbitals": 2}}]}, "do not fit"),
    ({"analyses": [{"name": "a", "modes": ["approx"]}]}, "bound_extension"),
    ({"analyses": [{"name": "a", "modes": ["below"]}]}, "energy_threshold"),
    ({"analyses": [{"name": "a", "modes": ["manual"]}]}, "not sweepable"),
    ({"analyses": [{"name": "a", "poly_degrees": [50]}]}, "poly_degrees"),
    ({"analyses": [{"name": "a", "recipe_overrides": {"target_indices": [0]}}]}, "filled in by the sweep"),
    ({"analyses": [{"name": "a", "target": {"strategy": "index"}}]}, "integer 'index'"),
    ({"analyses": [{"name": "a"}, {"name": "a", "poly_degrees": [2]}]}, "unique"),
])
def test_invalid_configs_fail_before_running(overrides, message):
    with pytest.raises(ConfigError, match=message):
        parse_sweep_config(minimal_config(**overrides))
