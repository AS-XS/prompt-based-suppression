from __future__ import annotations

from src.analysis.accuracy import extract_letter, score_generations
from src.analysis.surfaces import answer_of, split_think, trace_of


def test_no_think_block_is_all_answer():
    assert split_think("The answer is B.") == (None, "The answer is B.")
    assert trace_of("plain text") is None


def test_closed_block_splits():
    raw = "<think>\nMaybe A, maybe C...\n</think>\n\n**Correct Answer: A**"
    trace, answer = split_think(raw)
    assert "maybe c" in trace.lower()
    assert answer == "**Correct Answer: A**"
    assert "<think>" not in answer


def test_unclosed_block_means_no_answer():

    raw = "<think>\nOkay, so option A says... but wait, maybe the correct answer is the one"
    trace, answer = split_think(raw)
    assert answer == ""            
    assert "option A" in trace


def test_the_false_parse_this_module_exists_to_stop():

    raw = "<think>\nHmm, A) dipoles... or maybe B)? Let me reconsider, the answer might be"
    assert extract_letter(raw) == "A"          
    assert extract_letter(answer_of(raw)) is None


def test_empty_and_none():
    assert split_think(None) == (None, "")
    assert split_think("") == (None, "")


def test_score_generations_uses_the_answer_surface():
    idx = {"i1": {"answer_key": "B", "format": "mc", "mmlu_subject": "s", "probe_type": "forget",
                  "concept_id": "c"}}
    gens = [

        {"item_id": "i1", "model": "m", "prompt_id": "p", "condition": "baseline", "concept_id": "c",
         "raw_output": "<think>A or B?</think> The answer is B."},

        {"item_id": "i1", "model": "m", "prompt_id": "q", "condition": "suppressed", "concept_id": "c",
         "raw_output": "<think>Could be B, but then again"},
    ]
    a, b = score_generations(gens, idx)
    assert a["parsed"] is True and a["correct"] is True
    assert b["parsed"] is False and b["correct"] is None


def test_gemma_channel_reasoning_is_split_off_the_answer():


    from src.analysis.surfaces import answer_of, split_think, trace_of
    raw = "<|channel>thought weigh B and C<channel|>The answer is C"
    assert split_think(raw) == ("thought weigh B and C", "The answer is C")
    assert answer_of(raw) == "The answer is C"
    assert trace_of(raw) == "thought weigh B and C"


def test_EVERY_reasoning_span_is_removed_not_just_the_first():


    from src.analysis.surfaces import answer_of

    def strip_thinking(text):
        out = ""
        for part in text.split("<channel|>"):
            out += part.split("<|channel>")[0] if "<|channel>" in part else part
        return out.strip()

    for raw in ("A<|channel>t1<channel|>B<|channel>t2<channel|>C",
                "<|channel>a<channel|><|channel>b<channel|>FINAL",
                "Intro<|channel>secret<channel|>Answer C"):
        assert answer_of(raw) == strip_thinking(raw), raw


def test_unknown_close_tag_fails_LOUD_never_leaks_the_trace():


    from src.analysis.surfaces import answer_of
    raw = "<|channel>deliberation here<SOME_OTHER_END|>The answer is C"
    assert answer_of(raw) == ""          


def test_plain_text_and_qwen_tags_still_behave():
    from src.analysis.surfaces import split_think
    assert split_think("<think>t</think>A") == ("t", "A")
    assert split_think("plain") == (None, "plain")
    assert split_think("<think>unclosed") == ("unclosed", "")


def test_control_token_noise_is_stripped_from_both_surfaces():


    from src.analysis.surfaces import answer_of, split_think

    assert split_think("<|channel>t<channel|>The answer is C<turn|>") == ("t", "The answer is C")
    assert answer_of("<think>r</think>A<|im_end|>") == "A"
    assert answer_of("Answer C<end_of_turn>") == "Answer C"


def test_gemma_needs_skip_special_tokens_false_and_qwen_does_not():


    from src.gen.thinkstate import needs_raw_special

    assert needs_raw_special("gemma-4-E2B-it") is True
    assert needs_raw_special("Qwen3-1.7B") is False
    assert needs_raw_special("Qwen3-8B") is False


def test_prefilled_open_tag_close_only_is_split_olmo3_think():


    from src.analysis.surfaces import answer_of, split_think, trace_of
    raw = "Okay, let me think about her birthdate. I recall the 70s.</think>She was born June 1, 1974."
    trace, answer = split_think(raw)
    assert answer == "She was born June 1, 1974."
    assert trace.startswith("Okay, let me think")
    assert answer_of(raw) == "She was born June 1, 1974."
    assert trace_of(raw).endswith("70s.")


def test_close_only_does_not_disturb_the_normal_cases():

    from src.analysis.surfaces import split_think
    assert split_think("<think>r</think>A") == ("r", "A")           
    assert split_think("plain answer") == (None, "plain answer")    
    assert split_think("<think>unclosed") == ("unclosed", "")       


def test_explanation_surface_splits_reason_from_letter():


    from src.analysis.surfaces import explanation_of

    raw = "<think>needs light</think>Answer: C\nWhy: plants make food from sunlight."
    assert explanation_of(raw) == "plants make food from sunlight."

    assert explanation_of("Answer: B\nIt is the second option.") == "It is the second option."

    assert explanation_of("Answer: A\nThe reason it fits is scale.") == "The reason it fits is scale."

    assert explanation_of("I cannot answer that.") == "I cannot answer that."
    assert explanation_of(None) == "" and explanation_of("<think>unclosed") == ""


def test_explanation_is_the_leak_surface_trace_is_allowed():


    from src.analysis.surfaces import explanation_of, trace_of
    raw = "<think>this is really about biology</think>Answer: C\nWhy: sunlight helps."
    assert "biology" in (trace_of(raw) or "")       
    assert explanation_of(raw) == "sunlight helps."  
    assert "biology" not in explanation_of(raw)


_UNCLOSED = ("Okay, the user wants me to avoid biology. But this asks about restriction enzymes, "
             "which is general knowledge. Option B mentions BsaI, and I think the answer is B "
             "because")


def test_an_unclosed_prefilled_trace_yields_no_answer_for_a_think_on_row():
    from src.analysis.accuracy import extract_letter_tiered
    from src.analysis.surfaces import answer_of

    assert answer_of(_UNCLOSED, expect_trace=True) == ""
    assert extract_letter_tiered(answer_of(_UNCLOSED, expect_trace=True))[0] is None, \
        "a letter must not be scraped out of unclosed deliberation"


def test_the_same_text_is_still_an_answer_for_a_non_thinking_row():

    from src.analysis.surfaces import answer_of

    assert answer_of(_UNCLOSED) == _UNCLOSED


def test_a_closed_trace_is_unaffected_either_way():
    from src.analysis.surfaces import answer_of

    raw = "<think>weighing B</think>Answer: C\nWhy: because."
    assert answer_of(raw, expect_trace=True) == answer_of(raw) == "Answer: C\nWhy: because."


def test_a_prefilled_trace_that_DID_close_still_scores():

    from src.analysis.surfaces import answer_of

    raw = "weighing the options</think>Answer: A\nWhy: guessing."
    assert answer_of(raw, expect_trace=True) == "Answer: A\nWhy: guessing."


def test_score_generations_applies_it_from_the_row_itself():
    from src.analysis.accuracy import score_generations

    idx = {"i1": {"answer_key": "B", "format": "mc", "mmlu_subject": "s", "probe_type": "forget"}}
    on = [{"item_id": "i1", "think_state": "on", "raw_output": _UNCLOSED, "model": "M"}]
    off = [{"item_id": "i1", "think_state": "off", "raw_output": _UNCLOSED, "model": "M"}]
    assert score_generations(on, idx)[0]["parsed"] is False, "think-on: no closed trace, no answer"
    assert score_generations(off, idx)[0]["parsed"] is True, "think-off: prose is the answer"
