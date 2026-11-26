from minnow.fanout import CACHE_GIGABYTES, Fanout, by_round

# Qwen2.5-Coder-0.5B in half precision: keys and values, 24 layers, 2 heads of 64 numbers
TOKEN_BYTES = 2 * 24 * 2 * 64 * 2


def test_everything_goes_in_one_pass_when_it_fits():
    plan = Fanout(copies=14, max_new_tokens=256, temperature=0.8, seed=16000)

    assert plan.passes(prompts=16, prompt_tokens=300, token_bytes=TOKEN_BYTES) == [(16001, 14)]


def test_long_prompts_are_decoded_in_even_passes_that_fit_the_budget():
    plan = Fanout(copies=14, max_new_tokens=256, temperature=0.8, seed=0)

    passes = plan.passes(prompts=16, prompt_tokens=1000, token_bytes=TOKEN_BYTES)

    assert passes == [(1, 7), (8, 7)]
    for _, copies in passes:
        assert 16 * copies * (1000 + 256) * TOKEN_BYTES <= CACHE_GIGABYTES * 1e9


def test_uneven_split_gives_the_first_passes_one_copy_more():
    plan = Fanout(copies=5, max_new_tokens=10, temperature=None, seed=100, budget_bytes=2000)

    passes = plan.passes(prompts=1, prompt_tokens=90, token_bytes=10)

    assert passes == [(101, 2), (103, 2), (105, 1)]


def test_one_copy_goes_through_even_when_it_is_over_the_budget():
    plan = Fanout(copies=3, max_new_tokens=256, temperature=0.8, seed=0, budget_bytes=1)

    passes = plan.passes(prompts=16, prompt_tokens=4000, token_bytes=TOKEN_BYTES)

    assert passes == [(1, 1), (2, 1), (3, 1)]


def test_no_copies_means_no_passes():
    plan = Fanout(copies=0, max_new_tokens=256, temperature=0.8, seed=0)

    assert plan.passes(prompts=16, prompt_tokens=500, token_bytes=TOKEN_BYTES) == []


def test_every_pass_seed_is_different_and_all_copies_are_written():
    plan = Fanout(copies=29, max_new_tokens=256, temperature=0.8, seed=32000)

    passes = plan.passes(prompts=16, prompt_tokens=1800, token_bytes=TOKEN_BYTES)

    assert sum(copies for _, copies in passes) == 29
    assert len({seed for seed, _ in passes}) == len(passes)
    assert all(32000 < seed < 33000 for seed, _ in passes)


def test_answers_listed_prompt_by_prompt_become_rounds():
    answers = ["a1", "a2", "a3", "b1", "b2", "b3"]

    assert by_round(answers, copies=3) == [["a1", "b1"], ["a2", "b2"], ["a3", "b3"]]
