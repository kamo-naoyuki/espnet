import pytest

from espnet2.samplers.folded_batch_sampler import FoldedBatchSampler


@pytest.fixture()
def shape_files(tmp_path):
    p1 = tmp_path / "shape1.txt"
    with p1.open("w") as f:
        f.write("a 1000,80\n")
        f.write("b 400,80\n")
        f.write("c 800,80\n")
        f.write("d 789,80\n")
        f.write("e 1023,80\n")
        f.write("f 999,80\n")

    p2 = tmp_path / "shape2.txt"
    with p2.open("w") as f:
        f.write("a 30,30\n")
        f.write("b 50,30\n")
        f.write("c 39,30\n")
        f.write("d 49,30\n")
        f.write("e 44,30\n")
        f.write("f 99,30\n")

    return str(p1), str(p2)


@pytest.fixture()
def filtered_key_file(tmp_path):
    p1 = tmp_path / "filtered_keys.txt"
    with p1.open("w") as f:
        f.write("a\n")
        f.write("f\n")
    return str(p1)


@pytest.mark.parametrize("sort_in_batch", ["descending", "ascending"])
@pytest.mark.parametrize("sort_batch", ["descending", "ascending"])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("filtered", [True, False])
def test_FoldedBatchSampler(
    shape_files, sort_in_batch, sort_batch, drop_last, filtered_key_file, filtered
):
    sampler = FoldedBatchSampler(
        2,
        shape_files=shape_files,
        fold_lengths=[500, 80],
        sort_in_batch=sort_in_batch,
        sort_batch=sort_batch,
        drop_last=drop_last,
        filtered_key_file=filtered_key_file if filtered else None,
    )
    list(sampler)
    print(sampler)
    if filtered:
        assert len(sampler) == 3
    else:
        assert len(sampler) == 5
