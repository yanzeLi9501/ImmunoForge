"""Tests for immunoforge.core.affinity module."""

import pytest
from immunoforge.core.affinity import (
    AffinityResult,
    estimate_contacts,
    prodigy_binding_seq,
    rosetta_ref2015,
    bsa_regression,
    hotspot_score,
    consensus_kd,
    run_affinity_analysis,
    classify_binder_type,
    identify_cdrs,
    paratope_bsa_estimate,
)

SEQ = "MAELIKSYGVSWAIDGPISLNAKLTYLGFHPN"


class TestProdigyBindingSeq:
    def test_returns_affinity_result(self):
        contacts = estimate_contacts(SEQ, bsa=1200)
        result = prodigy_binding_seq(contacts, bsa=1200, binder_seq=SEQ)
        assert isinstance(result, AffinityResult)
        assert result.kd_nM > 0
        assert result.kd_nM < 5e8  # should NOT hit ceiling

    def test_larger_bsa_stronger_binding(self):
        c1 = estimate_contacts(SEQ, bsa=800)
        c2 = estimate_contacts(SEQ, bsa=1600)
        r1 = prodigy_binding_seq(c1, bsa=800, binder_seq=SEQ)
        r2 = prodigy_binding_seq(c2, bsa=1600, binder_seq=SEQ)
        # Larger BSA -> more contacts -> lower (more negative) ΔG
        assert r2.dg_kcal_mol < r1.dg_kcal_mol


class TestRosettaREF2015:
    def test_returns_affinity_result(self):
        contacts = estimate_contacts(SEQ, bsa=1200)
        result = rosetta_ref2015(contacts, bsa=1200, sc=0.65)
        assert isinstance(result, AffinityResult)
        assert result.kd_nM >= 0


class TestBSARegression:
    def test_returns_affinity_result(self):
        result = bsa_regression(SEQ, bsa=1200)
        assert isinstance(result, AffinityResult)
        assert result.kd_nM > 0


class TestHotspot:
    def test_trp_residues_score(self):
        seq_with_trp = "MAELWWWAIDGPISLNAKLTYLGFHPN"
        result = hotspot_score(seq_with_trp)
        assert result["hotspot_score"] > 0
        assert result["n_hotspot_residues"] > 0

    def test_no_hotspot_residues(self):
        seq = "MAELAAAIDGPISLN"
        result = hotspot_score(seq)
        assert result["hotspot_score"] == 0


class TestConsensusKD:
    def test_consistent_methods(self):
        results = [
            AffinityResult(method="a", dg_kcal_mol=-10.0, kd_nM=100.0, details={}),
            AffinityResult(method="b", dg_kcal_mol=-10.1, kd_nM=120.0, details={}),
            AffinityResult(method="c", dg_kcal_mol=-10.05, kd_nM=110.0, details={}),
        ]
        result = consensus_kd(results)
        assert result["confidence"] == "high"
        assert 90 < result["consensus_kd_nM"] < 150

    def test_divergent_methods_low_confidence(self):
        results = [
            AffinityResult(method="a", dg_kcal_mol=-10.0, kd_nM=10.0, details={}),
            AffinityResult(method="b", dg_kcal_mol=-5.0, kd_nM=100000.0, details={}),
            AffinityResult(method="c", dg_kcal_mol=-7.0, kd_nM=100.0, details={}),
        ]
        result = consensus_kd(results)
        assert result["confidence"] == "low"


class TestConsensusExcludesCeiling:
    def test_ceiling_excluded(self):
        results = [
            AffinityResult(method="a", dg_kcal_mol=0, kd_nM=1e9, details={}),
            AffinityResult(method="b", dg_kcal_mol=-10, kd_nM=100.0, details={}),
            AffinityResult(method="c", dg_kcal_mol=-9, kd_nM=200.0, details={}),
        ]
        result = consensus_kd(results)
        assert result["consensus_kd_nM"] is not None
        assert result["consensus_kd_nM"] < 1000
        assert "a" in result["excluded_methods"]
        assert result["n_methods_used"] == 2


class TestDomainClassification:
    def test_vh_detected(self):
        # Nivolumab VH contains FR patterns
        vh = "QVQLVESGGGVVQPGRSLRLDCKASGITFSNSGMHWVRQAPGKGLEWVAVIWYDGSKRYYADSVKGRFTISRDNSKNTLFLQMNSLRAEDTAVYYCATNDDYWGQGTLVTVSS"
        assert classify_binder_type(vh) in ("VH", "scFv")

    def test_natural_protein(self):
        assert classify_binder_type(SEQ) == "natural_protein"

    def test_cdr_identification(self):
        vh = "QVQLVESGGGVVQPGRSLRLDCKASGITFSNSGMHWVRQAPGKGLEWVAVIWYDGSKRYYADSVKGRFTISRDNSKNTLFLQMNSLRAEDTAVYYCATNDDYWGQGTLVTVSS"
        cdrs = identify_cdrs(vh)
        # Should identify at least some CDRs
        assert isinstance(cdrs, dict)


class TestParatopeBSA:
    def test_no_cdrs_returns_total(self):
        assert paratope_bsa_estimate(SEQ, 1500.0, {}) == 1500.0

    def test_with_cdrs_returns_less(self):
        cdrs = {
            "CDR-H1": {"seq": "GYTFTS", "start": 26, "end": 32},
            "CDR-H3": {"seq": "ATNDDY", "start": 95, "end": 101},
        }
        result = paratope_bsa_estimate("A" * 110, 1500.0, cdrs)
        assert result < 1500.0


class TestRunAffinityAnalysis:
    def test_full_analysis(self):
        result = run_affinity_analysis(
            binder_seq=SEQ,
            bsa=1200,
            sc=0.65,
        )
        assert "prodigy" in result
        assert "rosetta" in result
        assert "bsa_reg" in result
        assert "consensus" in result
        assert "hotspot" in result
        assert "domain_classification" in result

    def test_prodigy_not_ceiling(self):
        result = run_affinity_analysis(binder_seq=SEQ, bsa=1500, sc=0.70)
        assert result["prodigy"]["kd_nM"] < 5e8

    def test_consensus_uses_valid_methods(self):
        result = run_affinity_analysis(binder_seq=SEQ, bsa=1200, sc=0.65)
        cons = result["consensus"]
        assert cons["consensus_kd_nM"] is not None
        assert cons["n_methods_used"] >= 2
