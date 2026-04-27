"""Tests for immunoforge.core.sequence_qc module."""

import pytest
from immunoforge.core.sequence_qc import run_full_qc, batch_qc


class TestRunFullQC:
    """Unit tests for single-sequence QC."""

    def test_clean_sequence_passes(self):
        seq = "MAELIKSYGVSWAIDGPISLNAKLTYLGFHPN"
        result = run_full_qc(seq)
        assert result["pass"] is True
        assert result["failures"] == []

    def test_furin_site_detected(self):
        # RxKR pattern
        seq = "MAELIKSYGVSRSKRAIDGPISLNAKLTYLGFH"
        result = run_full_qc(seq)
        assert result["protease"].get("furin", 0) > 0

    def test_poly_cationic_toxicity(self):
        seq = "MAELRRRRRRRRGPISLNAKLTYLGFHPN"
        result = run_full_qc(seq)
        assert result["pass"] is False

    def test_aggregation_prone_region(self):
        seq = "MAELIIIIIIIGPISLNAKLTYLGFHPN"
        result = run_full_qc(seq)
        assert result["aggregation"]["apr_count"] > 0

    def test_odd_cysteine_warning(self):
        seq = "MAELCKSYGVSWAIDGPISLNAKLTYLGFHPN"
        result = run_full_qc(seq)
        assert result["cysteine"]["n_cysteines"] == 1
        assert result["cysteine"]["is_even"] is False

    def test_very_short_sequence(self):
        seq = "MAELI"
        result = run_full_qc(seq)
        # Should still return a result dict
        assert "pass" in result
        assert "sequence_length" in result


class TestBatchQC:
    """Tests for batch QC processing."""

    def test_batch_returns_dict(self):
        seqs = [
            ("s1", "MAELIKSYGVSWAIDGPISLNAKLTYLGFHPN"),
            ("s2", "MAELI"),
        ]
        results = batch_qc(seqs)
        assert isinstance(results, dict)
        assert "total" in results
        assert results["total"] == 2

    def test_batch_mixed_pass_fail(self):
        seqs = [
            ("good", "MAELIKSYGVSWAIDGPISLNAKLTYLGFHPN"),
            ("bad", "MAELRRRRRRRRGPISLNAKLTYLGFHPN"),
        ]
        results = batch_qc(seqs)
        assert results["n_passed"] + results["n_failed"] == results["total"]
