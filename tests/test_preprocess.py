from textprep_bench.preprocess import PreprocessConfig, light_stem, preprocess

RAW = PreprocessConfig()


def test_raw_only_normalises_whitespace():
    assert preprocess("  Hello   World  ", RAW) == "Hello World"


def test_none_and_nan_become_empty():
    assert preprocess(None, RAW) == ""
    assert preprocess(float("nan"), RAW) == ""


def test_lowercase():
    assert preprocess("SAD Today", PreprocessConfig(lowercase=True)) == "sad today"


def test_urls_modes():
    t = "see https://t.co/abc123 now"
    assert "xxurl" in preprocess(t, PreprocessConfig(urls="token"))
    assert "t.co" not in preprocess(t, PreprocessConfig(urls="remove"))
    assert "t.co" in preprocess(t, PreprocessConfig(urls="keep"))


def test_mentions_and_hashtags():
    assert preprocess("hi @bob", PreprocessConfig(mentions="token")) == "hi xxuser"
    assert preprocess("hi @bob", PreprocessConfig(mentions="remove")) == "hi"
    assert preprocess("so #tired", PreprocessConfig(hashtags="strip")) == "so tired"
    assert preprocess("so #tired", PreprocessConfig(hashtags="remove")) == "so"
    assert preprocess("so #tired", PreprocessConfig(hashtags="keep")) == "so #tired"


def test_emoji_modes():
    t = "ok \U0001F61E\U0001F61E"
    assert preprocess(t, PreprocessConfig(emoji="token")) == "ok xxemoji"
    assert preprocess(t, PreprocessConfig(emoji="remove")) == "ok"
    assert "\U0001F61E" in preprocess(t, PreprocessConfig(emoji="keep"))


def test_elongation_reduces_runs_to_two():
    assert preprocess("sooooo tired", PreprocessConfig(elongation=True)) == "soo tired"


def test_punctuation_stripping_keeps_emoji_when_emoji_kept():
    out = preprocess("sad! \U0001F61E", PreprocessConfig(strip_punctuation=True))
    assert out == "sad \U0001F61E"


def test_stopwords_keep_or_drop_negations():
    keep = PreprocessConfig(lowercase=True, stopwords=True, keep_negations=True)
    drop = PreprocessConfig(lowercase=True, stopwords=True, keep_negations=False)
    assert preprocess("i am not happy", keep) == "not happy"
    assert preprocess("i am not happy", drop) == "happy"


def test_negation_marking_scope_and_reset():
    cfg = PreprocessConfig(negation_marking=True)
    assert preprocess("not very happy today ok", cfg) == "not not_very not_happy not_today ok"
    assert preprocess("not happy. fine", cfg) == "not not_happy . fine"
    assert preprocess("I don't like it", cfg) == "I don't not_like not_it"


def test_light_stemmer():
    assert light_stem("feeling") == "feel"
    assert light_stem("friends") == "friend"
    assert light_stem("happiness") == "happiness"  # -ss is protected
    assert light_stem("sad") == "sad"


def test_stemming_keeps_negation_prefix_and_placeholders():
    cfg = PreprocessConfig(negation_marking=True, stemming="light", urls="token")
    assert preprocess("not feeling https://x.co", cfg) == "not not_feel not_xxurl"


def test_min_token_len_exempts_punctuation():
    cfg = PreprocessConfig(min_token_len=2)
    assert preprocess("i am sad !", cfg) == "am sad !"


def test_deterministic_and_pure():
    cfg = PreprocessConfig(lowercase=True, negation_marking=True, stemming="light")
    text = "Not feeling GREAT today!!! #tired"
    assert preprocess(text, cfg) == preprocess(text, cfg)


def test_invalid_config_rejected():
    for bad in ({"urls": "oops"}, {"stemming": "x"}, {"min_token_len": 0}):
        try:
            PreprocessConfig(**bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should have been rejected")
