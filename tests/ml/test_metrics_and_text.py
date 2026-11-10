import pytest
import torch

from ml.data.text import get_tokenizer, normalize_text
from ml.evaluation.metrics import cer, char_errors, wer, word_errors


def test_normalize_text_canonical_form():
    assert normalize_text("  Put red, at G9 now!  ") == "PUT RED AT G9 NOW"
    assert normalize_text("it’s {LG} café") == "IT'S CAFE"


def test_wer_counts_match_hand_computation():
    # 6 reference words; two substitutions: red->blue, now->please
    ref = ["bin red at f two now"]
    hyp = ["bin blue at f two please"]
    counts = word_errors(ref, hyp)
    assert counts.reference_length == 6
    assert (counts.substitutions, counts.deletions, counts.insertions) == (2, 0, 0)
    assert wer(ref, hyp) == pytest.approx(2 / 6)


def test_wer_insertions_and_deletions():
    assert wer(["a b c d"], ["a b c d e f"]) == pytest.approx(2 / 4)  # 2 insertions
    assert wer(["a b c d"], ["a d"]) == pytest.approx(2 / 4)  # 2 deletions


def test_corpus_level_not_mean_of_utterances():
    refs = ["a", "b c d e f g h i j"]  # 1 + 9 words
    hyps = ["x", "b c d e f g h i j"]  # 1 error total
    assert wer(refs, hyps) == pytest.approx(1 / 10)  # mean of per-utt would be 0.5


def test_normalisation_applied_before_scoring():
    assert wer(["Hello, World"], ["hello world"]) == 0.0
    assert cer(["abc"], ["abd"]) == pytest.approx(1 / 3)
    assert char_errors(["ab cd"], ["ab cd"]).reference_length == 5  # spaces count


def test_metric_input_validation():
    with pytest.raises(ValueError):
        wer([], [])
    with pytest.raises(ValueError):
        wer(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        wer(["!!!"], ["a"])  # empty after normalisation


def test_tokenizer_matches_checkpoint_vocabulary():
    tok = get_tokenizer()
    assert tok.vocab_size == 5049  # <blank> + 5047 units + <eos>
    assert tok.token_list[0] == "<blank>" and tok.token_list[-1] == "<eos>"
    ids = tok.encode("place white by q seven again")
    assert ids.dtype == torch.long and (ids > 0).all() and (ids < tok.vocab_size - 1).all()
    assert tok.decode(ids) == "PLACE WHITE BY Q SEVEN AGAIN"
