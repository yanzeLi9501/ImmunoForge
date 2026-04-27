"""Tests for immunoforge.core.target_db module."""

import pytest
from immunoforge.core.target_db import (
    search_targets,
    get_target_by_name,
    list_cell_types,
    list_species,
    get_benchmark_targets,
)


class TestSearchTargets:
    def test_all_targets(self):
        targets = search_targets()
        assert len(targets) > 0

    def test_filter_by_cell_type(self):
        targets = search_targets(cell_type="T cell")
        assert all(t.cell_type == "T cell" for t in targets)

    def test_filter_by_species(self):
        targets = search_targets(species="mouse")
        assert all("mouse" in t.species.lower() or "mus" in t.species.lower() for t in targets)

    def test_filter_by_name(self):
        targets = search_targets(name="CD3")
        assert len(targets) > 0


class TestGetTargetByName:
    def test_existing_target(self):
        target = get_target_by_name("mCLEC9A")
        assert target is not None
        assert target.name == "mCLEC9A"

    def test_nonexistent_target(self):
        target = get_target_by_name("NONEXISTENT_XYZ")
        assert target is None


class TestListCellTypes:
    def test_returns_set(self):
        types = list_cell_types()
        assert isinstance(types, (set, list))
        assert "T cell" in types

    def test_tumour_cell_type(self):
        types = list_cell_types()
        assert "tumour" in types


class TestListSpecies:
    def test_returns_set(self):
        species = list_species()
        assert len(species) > 0


class TestBenchmarkTargets:
    def test_returns_benchmarks(self):
        benchmarks = get_benchmark_targets()
        assert len(benchmarks) > 0
        assert all(t.benchmark_value is not None for t in benchmarks)
