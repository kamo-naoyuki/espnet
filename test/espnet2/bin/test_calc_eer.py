from argparse import ArgumentParser

import numpy
import pytest

from espnet2.bin.calc_eer import get_parser
from espnet2.bin.calc_eer import main
from espnet2.bin.calc_eer import write_eer


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


def test_write_eer(tmp_path):
    write_eer(
        output_dir=tmp_path,
        log_level="INFO",
        label=numpy.array([1, 0]),
        score=numpy.array([0.2, 0.5]),
    )


def test_write_eer_zero_len(tmp_path):
    write_eer(
        output_dir=tmp_path,
        log_level="INFO",
        label=numpy.array([]),
        score=numpy.array([]),
    )


def test_write_eer_nan(tmp_path):
    write_eer(
        output_dir=tmp_path,
        log_level="INFO",
        label=numpy.array([1, 0]),
        score=numpy.array([numpy.nan, numpy.nan]),
    )
