"""Tests for immunoforge.core.ranking module."""

import pytest
from immunoforge.core.ranking import (
    compute_physicochemical,
    compute_composite_score,
    rank_candidates,
)


class TestPhysicochemical:
    def test_returns_properties(self):
        props = compute_physicochemical("MAELIKSYGVSWAIDGPISLNAKLTYLGFHPN")
        assert "mw_da" in props
        assert "pI" in props
        assert "hydrophobic_fraction" in props
        assert "length" in props

    def test_mw_positive(self):
        props = compute_physicochemical("MAELI")
        assert props["mw_da"] > 0

    def test_pi_in_range(self):
        props = compute_physicochemical("MAELI")
        assert 0 < props["pI"] < 14


class TestCompositeScore:
    def test_basic_scoring(self):
        physico = compute_physicochemical("MAELIKSYGVSWAIDGPISLNAKLTYLGFHPN")
        score = compute_composite_score(
            plddt=85.0, ddg=-5.0, kd_nM=50.0,
            mpnn_score=-1.5, bsa=1200, sc=0.65,
            physico=physico,
        )
        assert isinstance(score, float)
        assert score > 0

    def test_better_metrics_higher_score(self):
        physico = compute_physicochemical("MAELIKSYGVSWAIDGPISLNAKLTYLGFHPN")
        good = compute_composite_score(
            plddt=95.0, ddg=-10.0, kd_nM=10.0,
            mpnn_score=-2.5, bsa=1800, sc=0.80,
            physico=physico,
        )
        bad = compute_composite_score(
            plddt=60.0, ddg=-1.0, kd_nM=5000.0,
            mpnn_score=-0.5, bsa=600, sc=0.30,
            physico=physico,
        )
        assert good > bad


class TestRankCandidates:
    def test_returns_sorted(self):
        candidates = [
            {
                "id": f"c{i}",
                "sequence": "MAELIKSYGVSWAIDGPISLNAKLTYLGFHPN",
                "plddt": 70 + i * 5,
                "ddg": -3.0 - i,
                "kd_nM": 500 - i * 100,
                "mpnn_score": -1.0 - i * 0.3,
                "bsa": 1000 + i * 100,
                "sc": 0.5 + i * 0.05,
            }
            for i in range(5)
        ]
        ranked = rank_candidates(candidates, top_n=3)
        assert len(ranked) == 3
        scores = [r["composite_score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)
