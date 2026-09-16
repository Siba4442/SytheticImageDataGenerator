import argparse

import pytest

from visionsynth import cli


def test_int_pair_parses_min_max() -> None:
    assert cli._int_pair("10,25") == (10, 25)


def test_int_pair_rejects_malformed_input() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        cli._int_pair("not-a-pair")


def test_parse_arguments_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["visionsynth"])
    args = cli.parse_arguments()

    assert args.background == "plain"
    assert args.font is None
    assert args.output_dir == "output/"
    assert args.birch_spot_radius == (10, 25)
    assert args.parchment_spot_radius == (3, 12)


def test_main_requires_font(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr("sys.argv", ["visionsynth", "--input-csv", "unused.csv"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 1
    assert "--font is required" in capsys.readouterr().out
