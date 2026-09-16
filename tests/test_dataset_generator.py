import pathlib

import pytest

from visionsynth import dataset_generator

NON_IMAGE_BACKGROUNDS = ["plain", "gaussian", "lined", "old", "birch", "parchment"]


@pytest.mark.parametrize("background", NON_IMAGE_BACKGROUNDS)
def test_build_background_dispatches_by_style(background: str) -> None:
    params = {"width": 40, "height": 30, "background": background}

    image, style = dataset_generator._build_background(params)

    assert style == background
    assert image.size == (40, 30)


def test_build_background_image_requires_image_dir() -> None:
    params = {"width": 40, "height": 30, "background": "image"}

    with pytest.raises(ValueError, match="background_image_dir"):
        dataset_generator._build_background(params)


def test_build_background_defaults_to_plain() -> None:
    _, style = dataset_generator._build_background({"width": 10, "height": 10})
    assert style == "plain"


def test_load_dataset_drops_blank_and_missing_text_rows(tmp_path: pathlib.Path) -> None:
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text(
        'text,label\n"Hello world",a\n"",b\n,c\n"Another row",d\n', encoding="utf-8"
    )

    df = dataset_generator.load_dataset(str(csv_path), "text")

    assert df is not None
    assert list(df["text"]) == ["Hello world", "Another row"]


def test_load_dataset_missing_column_returns_none(tmp_path: pathlib.Path) -> None:
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("other_column\nvalue\n", encoding="utf-8")

    assert dataset_generator.load_dataset(str(csv_path), "text") is None


def test_load_dataset_missing_file_returns_none() -> None:
    assert dataset_generator.load_dataset("/nonexistent/path.csv", "text") is None
