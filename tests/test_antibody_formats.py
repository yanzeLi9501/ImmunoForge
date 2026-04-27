"""Tests for immunoforge.core.antibody_formats module."""

import pytest
from immunoforge.core.antibody_formats import (
    ANTIBODY_FORMATS,
    CROSS_SPECIES_IMMUNOGENICITY,
    SPECIES_VH_FRAMEWORKS,
    get_format,
    list_formats,
    get_immunogenicity,
    list_immunogenicity_matrix,
    compare_species_sequences,
    design_construct,
)


class TestFormatCatalogue:
    def test_catalogue_not_empty(self):
        assert len(ANTIBODY_FORMATS) >= 15

    def test_all_formats_have_name(self):
        for f in ANTIBODY_FORMATS:
            assert f.name
            assert f.format_type
            assert f.species

    def test_human_igg_subtypes(self):
        human_igg = list_formats(species="human", format_type="IgG")
        names = [f.name for f in human_igg]
        assert "Human IgG1" in names
        assert "Human IgG4" in names

    def test_mouse_formats(self):
        mouse = list_formats(species="mouse")
        assert len(mouse) >= 2
        names = [f.name for f in mouse]
        assert "Mouse IgG1" in names
        assert "Mouse IgG2a" in names

    def test_nanobody_formats(self):
        nb = list_formats(format_type="nanobody")
        assert len(nb) >= 2

    def test_bispecific_formats(self):
        bi = list_formats(format_type="bispecific")
        assert len(bi) >= 3
        names = [f.name for f in bi]
        assert "KiH IgG bispecific" in names
        assert "Bispecific nanobody" in names

    def test_cynomolgus_format(self):
        cyno = list_formats(species="cynomolgus")
        assert len(cyno) >= 1
        assert cyno[0].name == "Cyno IgG4"


class TestGetFormat:
    def test_existing(self):
        f = get_format("Human IgG1")
        assert f is not None
        assert f.molecular_weight_kda == 146.0
        assert len(f.ch_sequence) > 100

    def test_nonexistent(self):
        assert get_format("Nonexistent_XYZ") is None

    def test_kih_mutations(self):
        f = get_format("KiH IgG bispecific")
        assert f is not None
        assert "T366W" in f.mutations


class TestImmunogenicity:
    def test_human_in_human(self):
        entry = get_immunogenicity("Human IgG1", "human")
        assert entry is not None
        assert entry.risk_level == "very_low"
        assert entry.risk_score < 0.1

    def test_mouse_in_human(self):
        entry = get_immunogenicity("Mouse IgG1", "human")
        assert entry is not None
        assert entry.risk_level == "high"
        assert entry.ada_incidence_pct > 50

    def test_mouse_in_mouse(self):
        entry = get_immunogenicity("Mouse IgG1", "mouse")
        assert entry is not None
        assert entry.risk_level == "very_low"

    def test_matrix_length(self):
        matrix = list_immunogenicity_matrix()
        assert len(matrix) == len(CROSS_SPECIES_IMMUNOGENICITY)
        assert all("format" in row for row in matrix)


class TestSpeciesComparison:
    def test_human_vs_mouse(self):
        result = compare_species_sequences("human", "mouse")
        assert "identity" in result
        assert 0.3 < result["identity"] < 1.0
        assert result["n_differences"] > 0

    def test_human_vs_camelid(self):
        result = compare_species_sequences("human", "camelid")
        assert "identity" in result

    def test_human_vs_cyno(self):
        result = compare_species_sequences("human", "cyno")
        assert "identity" in result
        # Cyno IGHV3 shares high but not 1:1 identity with human IGHV1
        # (comparison is human_IGHV1 vs cyno_IGHV3; CDR placeholders reduce score)
        assert result["identity"] > 0.50

    def test_cyno_immunogenicity(self):
        entry = get_immunogenicity("Cyno IgG4", "human")
        assert entry is not None
        assert entry.risk_level == "low"
        entry2 = get_immunogenicity("Human IgG1", "cynomolgus")
        assert entry2 is not None
        assert entry2.risk_score < 0.10

    def test_unknown_species(self):
        result = compare_species_sequences("unknown", "mouse")
        assert "error" in result


class TestDesignConstruct:
    VH_EXAMPLE = "QVQLVQSGAEVKKPGASVKVSCKASGYTFT"
    VL_EXAMPLE = "DIQMTQSPSSLSASVGDRVTITCRAS"

    def test_igg_construct(self):
        result = design_construct(self.VH_EXAMPLE, self.VL_EXAMPLE, "Human IgG1")
        assert "error" not in result
        assert result["format"] == "Human IgG1"
        assert len(result["full_sequence"]) > len(self.VH_EXAMPLE)
        assert "VH" in result["domains"]
        assert "CH" in result["domains"]

    def test_scfv_construct(self):
        result = design_construct(self.VH_EXAMPLE, self.VL_EXAMPLE, "Human scFv")
        assert "linker" in result["domains"]
        assert "GGGGS" in result["full_sequence"]

    def test_nanobody_construct(self):
        result = design_construct(self.VH_EXAMPLE, None, "Camelid VHH")
        assert result["full_sequence"] == self.VH_EXAMPLE
        assert "VHH" in result["domains"]

    def test_unknown_format(self):
        result = design_construct(self.VH_EXAMPLE, None, "Unknown_XYZ")
        assert "error" in result
