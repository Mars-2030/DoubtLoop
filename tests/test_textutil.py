"""The support check is the metric. If it drifts, every number in the repo does."""

import pytest

from groundloop import textutil as T


class TestSentenceSplitting:
    def test_keeps_abbreviations_and_initials_intact(self):
        got = T.split_sentences("Dr. Vandermeer runs KS-9. J. J. R. Macleod ran the lab.")
        assert got == ["Dr. Vandermeer runs KS-9.", "J. J. R. Macleod ran the lab."]

    def test_year_at_end_of_sentence_is_not_a_list_marker(self):
        # Regression: "1822." was treated as an enumeration marker, which glued
        # the next sentence on and hid a claim from the claim splitter.
        got = T.split_sentences("It was deciphered in 1822. I am not certain.")
        assert len(got) == 2

    def test_list_markers_at_the_head_stay_attached(self):
        assert T.split_sentences("1. First item. 2. Second item.") == ["1. First item.", "2. Second item."]


class TestClaimExtraction:
    def test_drops_questions_and_meta_commentary(self):
        claims = T.extract_claims("Based on the evidence, this holds. Is that right? The day is 31 hours.")
        assert claims == ["The day is 31 hours."]

    def test_abstention_is_not_a_claim(self):
        assert T.extract_claims("The corpus does not say. Nothing in the corpus states it.") == []

    def test_hedges_are_stripped_not_honoured(self):
        # A hedge must not erase the claim: otherwise the plain-critique
        # condition, whose whole strategy is hedging, scores as faithful.
        claims = T.extract_claims("As far as I recall: The station day is 24 hours.")
        assert claims == ["The station day is 24 hours."]

    def test_leading_citation_carried_over_by_the_splitter_is_removed(self):
        claims = T.extract_claims("The day is 31 hours. [f03] Water is reclaimed at 97.3 percent.")
        assert claims[-1].startswith("Water")


class TestCoverage:
    def test_verbatim_quote_is_fully_covered(self):
        text = "Michael Collins remained in lunar orbit aboard the command module Columbia."
        assert T.coverage(text, text) == pytest.approx(1.0)

    def test_citations_do_not_deflate_coverage(self):
        passage = "The KS-9 station day is 31 standard hours long."
        assert T.coverage("The KS-9 station day is 31 standard hours long. [f03]", passage) == pytest.approx(1.0)

    def test_rare_tokens_weigh_more_than_common_ones(self):
        # Sharing the common words is not support when the distinguishing ones
        # are missing.
        claim = "The Sable hull coating was developed in-house by the Consortium's own materials lab."
        passage = "Halden Materials is the Consortium supplier that developed the Sable coating."
        assert T.coverage(claim, passage) < 0.6


class TestCheckClaim:
    EV = [("f03", "The KS-9 station day is 31 standard hours long, chosen to match Aral's rotation.")]

    def test_supported_claim(self):
        ok, pid, _, _ = T.check_claim_default("The KS-9 station day is 31 standard hours long.", self.EV)
        assert ok and pid == "f03"

    def test_wrong_number_is_vetoed(self):
        ok, _, _, reason = T.check_claim_default("The KS-9 station day is 24 standard hours long.", self.EV)
        assert not ok and "numeric" in reason

    def test_a_number_the_sentence_merely_omits_is_not_a_disagreement(self):
        # Regression, found on the first real-model run: a correct two-part
        # answer was scored as a hallucination because the sentence carrying the
        # first date says nothing about the second - the next sentence does.
        # Treating a missing number as a contradiction rejects most claims that
        # span two sentences, which inflates the hallucination rate for any
        # model that writes full answers.
        evidence = [("r03", "The Rosetta Stone was found near Rashid in Egypt in 1799 by French "
                            "soldiers. Jean-Francois Champollion announced his decipherment in 1822.")]
        ok, _, _, _ = T.check_claim_default(
            "The Rosetta Stone was found in 1799 and deciphered by Champollion in 1822.", evidence)
        assert ok

    def test_an_invented_number_is_still_caught(self):
        evidence = [("f05", "Dr. Ilse Vandermeer has been the director of KS-9 since 2201.")]
        ok, _, _, _ = T.check_claim_default("Vandermeer was appointed director in 2207.", evidence)
        assert not ok

    def test_polarity_flip_is_vetoed(self):
        ev = [("r02", "The Calvin cycle runs in the stroma; it does not itself require light.")]
        ok, _, _, reason = T.check_claim_default("The Calvin cycle does require light.", ev)
        assert not ok and "polarity" in reason

    def test_no_evidence_means_no_support(self):
        ok, _, _, reason = T.check_claim_default("Anything at all.", [])
        assert not ok and reason == "no evidence available"

    def test_scattered_words_across_passages_do_not_add_up_to_support(self):
        ev = [("a", "Aral has fourteen confirmed moons."), ("b", "The greenhouse ring supplies calories.")]
        ok, _, _, _ = T.check_claim_default("The greenhouse ring orbits fourteen moons of Aral.", ev)
        assert not ok


def test_numbers_normalise_thousands_separators():
    assert T.numbers("Everest is 8,848.86 metres") == {"8848.86"}


class TestNumericDisagreement:
    """Both sides have to commit for it to be a disagreement."""

    def test_different_values_disagree(self):
        assert T.numeric_disagreement("the day is 24 hours", "the day is 31 hours")

    def test_a_superset_claim_does_not_disagree(self):
        assert not T.numeric_disagreement("found in 1799, deciphered in 1822", "found in 1799")

    def test_a_superset_source_does_not_disagree(self):
        assert not T.numeric_disagreement("found in 1799", "found in 1799, deciphered in 1822")

    def test_no_numbers_on_either_side(self):
        assert not T.numeric_disagreement("the sky is blue", "the day is 31 hours")
        assert not T.numeric_disagreement("the day is 31 hours", "the sky is blue")


def test_abstention_detection():
    assert T.is_abstention("The corpus does not say.")
    assert not T.is_abstention("The station day is 31 hours.")


class TestStemming:
    """Four separate false negatives on real-model output came from one word
    appearing in different forms in the claim and in the passage."""

    def test_inflections_and_one_derivation_collapse(self):
        assert T.stem("deciphered") == T.stem("decipherment") == "decipher"
        assert T.stem("membranes") == T.stem("membrane")
        assert T.stem("requires") == T.stem("require")

    def test_it_does_not_invent_matches(self):
        # An invented match is the expensive direction: it turns "the model is
        # wrong" into "the scorer cannot read".
        assert T.stem("foundation") != T.stem("found")
        assert T.stem("transistor") != T.stem("transit")
        assert T.stem("class") == "class", "'ss' must survive plural stripping"

    def test_short_words_are_left_alone(self):
        for word in ("gas", "is", "was", "ion"):
            assert T.stem(word) == word

    def test_a_correct_two_sentence_answer_is_supported(self):
        # The claim spans two sentences of one passage and uses "deciphered"
        # where the passage says "decipherment"; it scored 0.787 against a 0.80
        # bar before stemming.
        evidence = [("r03", "The Rosetta Stone was found near Rashid in Egypt in 1799 by French "
                            "soldiers. It carries a decree issued at Memphis in 196 BC on behalf of "
                            "Ptolemy V. Jean-Francois Champollion announced his decipherment of the "
                            "hieroglyphic text in 1822.")]
        ok, _, _, _ = T.check_claim_default(
            "The Rosetta Stone was found in 1799 and deciphered in 1822.", evidence)
        assert ok

    def test_grounding_is_not_correctness(self):
        # "deciphered in 1799" is factually wrong, but every number it states
        # does appear in the passage, so a grounding check cannot catch it -
        # the reference-answer check is what does. Conflating the two would
        # make the grounding metric unfalsifiable.
        evidence = [("r03", "The Rosetta Stone was found near Rashid in Egypt in 1799 by French "
                            "soldiers. Champollion announced his decipherment in 1822.")]
        ok, _, _, _ = T.check_claim_default(
            "The Rosetta Stone was found in 1799 and deciphered in 1799.", evidence)
        assert ok, "the grounding check is not expected to catch a role-swapped number"
