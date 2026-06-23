from eval.gold import GoldItem, save, load


def test_save_load_roundtrip(tmp_path):
    items = [GoldItem("Q?", "A.", ["d:1:0"], [1])]
    p = tmp_path / "g.jsonl"
    save(items, p)
    back = load(p)
    assert back == items
