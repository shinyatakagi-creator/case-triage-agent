from scripts.tally_template import lines, main


def test_one_blank_line_per_issue():
    assert lines(["141240", "135321"]) == ["141240 ?  <doc path>  (<evidence>)", "135321 ?  <doc path>  (<evidence>)"]


def test_main_prints_and_writes_nothing(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["1", "2"]) == 0
    assert capsys.readouterr().out == "1 ?  <doc path>  (<evidence>)\n2 ?  <doc path>  (<evidence>)\n"
    assert list(tmp_path.iterdir()) == []


def test_main_rejects_non_numbers(capsys):
    assert main(["12x"]) == 2 and main([]) == 2
