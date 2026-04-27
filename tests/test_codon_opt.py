"""Tests for immunoforge.core.codon_opt module."""

import pytest
from immunoforge.core.codon_opt import (
    optimize_codons,
    remove_t5nt,
    check_restriction_sites,
    build_expression_cassette,
    full_codon_optimization,
)


class TestOptimizeCodons:
    def test_correct_length(self):
        protein = "MAEL"
        dna = optimize_codons(protein, species="mouse")
        assert len(dna) == len(protein) * 3

    def test_translates_back(self):
        from immunoforge.core.utils import CODON_TABLE
        protein = "MAELIKY"
        dna = optimize_codons(protein, species="mouse")
        inv_table = {}
        for codon, aa in CODON_TABLE.items():
            inv_table.setdefault(aa, []).append(codon)
        for i in range(len(protein)):
            codon = dna[i * 3 : i * 3 + 3]
            assert CODON_TABLE.get(codon) == protein[i], f"Codon {codon} != {protein[i]}"

    def test_human_species(self):
        dna = optimize_codons("MAEL", species="human")
        assert len(dna) == 12

    def test_cynomolgus_species(self):
        dna = optimize_codons("MAEL", species="cynomolgus")
        assert len(dna) == 12


class TestRemoveT5NT:
    def test_no_change_if_no_t5nt(self):
        dna = "ATGCGATCGATCG"
        result = remove_t5nt(dna, "MAEL")
        # Should be unchanged if no TTTTTNT pattern
        assert len(result) == len(dna)


class TestRestrictionSites:
    def test_clean_sequence(self):
        dna = "ATGATGATGATG"
        sites = check_restriction_sites(dna)
        assert isinstance(sites, dict)

    def test_ecori_detected(self):
        dna = "ATGGAATTCATG"
        sites = check_restriction_sites(dna)
        assert "EcoRI" in sites


class TestExpressionCassette:
    def test_vaccinia_cassette(self):
        result = build_expression_cassette("ATGATGATG", system="vaccinia")
        assert "cassette_dna" in result
        assert result["cassette_length_bp"] > 9

    def test_aav_cassette(self):
        result = build_expression_cassette("ATGATGATG", system="aav")
        assert "cassette_dna" in result


class TestFullOptimization:
    def test_mouse_vaccinia(self):
        result = full_codon_optimization(
            protein_seq="MAELIKSYGVSWAIDGPISLNAKLTYLGFHPN",
            species="mouse",
            system="vaccinia",
        )
        assert "cds_dna" in result
        assert "cassette" in result
        assert result["species"] == "mouse"
        assert result["expression_system"] == "vaccinia"
        assert 0.3 < result["gc_content_cds"] < 0.8

    def test_human_aav(self):
        result = full_codon_optimization(
            protein_seq="MAELIKSYGVS",
            species="human",
            system="aav",
        )
        assert result["species"] == "human"
