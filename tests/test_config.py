from config import MATRIX, ExperimentConfig, EMBED_MODELS


def test_matrix_is_2x2():
    assert len(MATRIX) == 4
    names = {c.collection_name for c in MATRIX}
    assert names == {
        "fixed__bge_small", "fixed__e5_large",
        "semantic__bge_small", "semantic__e5_large",
    }


def test_embed_models_are_the_chosen_pair():
    assert EMBED_MODELS["bge_small"] == "BAAI/bge-small-en-v1.5"
    assert EMBED_MODELS["e5_large"] == "intfloat/e5-large-v2"
