from eval.run_matrix import to_markdown


def test_to_markdown_shapes_table():
    rows = [{"a": 1, "b": 2}]
    md = to_markdown(rows, ["a", "b"])
    assert md.splitlines()[0] == "| a | b |"
    assert md.splitlines()[2] == "| 1 | 2 |"
