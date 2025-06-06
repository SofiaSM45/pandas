import re

import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm


def test_error_unequal_lengths():
    # Raise Value Error when lists in same row have different lengths
    df = pd.DataFrame({"A": [[1, 2], [3]], "B": [["a", "b", "c"], ["d"]]})
    with pytest.raises(
        ValueError, match="Row 0 has unequal lengths in exploded columns"
    ):
        df.explode_wide(["A", "B"])


def test_error_empty_column():
    # Raise Value Error when no column is given to explode
    df = pd.DataFrame({"A": [[1, 2], [3]], "B": [["a", "b"], ["d"]]})
    with pytest.raises(ValueError, match="column must be nonempty"):
        df.explode_wide([])


def test_error_invalid_column_format():
    # Raise Value Error when input format is not a valid column(s)
    df = pd.DataFrame({"A": [[1, 2], [3]], "B": [["a", "b"], ["d"]]})
    with pytest.raises(
        ValueError, match="column must be a scalar, tuple, or list thereof"
    ):
        df.explode_wide({"A": 1})


def test_error_duplicate_columns():
    # Raise Value Error if DataFrame has duplicate column names
    df = pd.DataFrame(
        [
            [[1, 2], [3]],
            [["a", "b", "c"], ["d"]],
        ],
        columns=["A", "A"],
    )
    with pytest.raises(
        ValueError,
        match=re.escape("DataFrame columns must be unique. Duplicate columns: ['A']"),
    ):
        df.explode_wide("A")


def test_error_duplicate_column_input():
    # Raise Value Error when duplicate column is given to explode
    df = pd.DataFrame({"A": [[1, 2], [3]], "B": [["a", "b"], ["d"]]})
    with pytest.raises(ValueError, match="column must be unique"):
        df.explode_wide(["A", "A"])


def test_single_column():
    # Basic Case: single column of list, tuple exploded into separate columns
    df = pd.DataFrame({"A": [[0, 1, 2], (3, 4)], "B": 1})
    result = df.explode_wide("A")
    expected = pd.DataFrame(
        [
            [0, 1, 2, 3, 4],
            [1, 1, 1, 1, 1],
        ],
        index=["A", "B"],
        columns=[0, 0, 0, 1, 1],
    )
    tm.assert_frame_equal(result, expected)


def test_scalar_values():
    # Tests explode_wide with a mix of scalar values and lists
    df = pd.DataFrame({"A": [[1, 2], 3], "B": [["a", "b"], "c"]})
    result = df.explode_wide(["A", "B"])
    expected = pd.DataFrame(
        [[1, 2, 3], ["a", "b", "c"]], index=["A", "B"], columns=[0, 0, 1]
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("columns", ["A", ["A"]])
def test_input_column_variants(columns):
    # Accept both string and list input
    df = pd.DataFrame({"A": [[0, 1], [2, 3]], "B": [4, 5]})
    result = df.explode_wide(columns)

    expected = pd.DataFrame(
        [
            [0, 1, 2, 3],
            [4, 4, 5, 5],
        ],
        index=["A", "B"],
        columns=[0, 0, 1, 1],
    )
    tm.assert_frame_equal(result, expected)


def test_input_column_tuple_variant():
    # Explode column with MultiIndex column label (tuple)
    df = pd.DataFrame({("A", "B"): [[0, 1], [2, 3]], ("C", "D"): [[4, 5], [6, 7]]})
    result = df.explode_wide(("A", "B"))

    expected_index = pd.MultiIndex.from_tuples(
        [("A", "B"), ("C", "D")], names=[None, None]
    )
    expected = pd.DataFrame(
        [
            [0, 1, 2, 3],
            [[4, 5], [4, 5], [6, 7], [6, 7]],
        ],
        index=expected_index,
        columns=[0, 0, 1, 1],
    )
    tm.assert_frame_equal(result, expected)


def test_nan_in_lists():
    # Tests behavior when lists contain NaN values
    df = pd.DataFrame({"A": [[np.nan, 2], [3, np.nan]], "B": [1, 2]}, dtype=object)
    result = df.explode_wide("A").astype(object)

    expected = pd.DataFrame(
        [[np.nan, 2, 3, np.nan], [1, 1, 2, 2]],
        index=["A", "B"],
        columns=[0, 0, 1, 1],
        dtype=object,
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "explode_subset, expected_dict, expected_index, expected_columns",
    [
        (
            ["A"],
            [
                [0, 1, "foo", np.nan, 3, 4],
                [["a", "b"], ["a", "b"], np.nan, [], ["d", "e"], ["d", "e"]],
            ],
            ["A", "B"],
            [0, 0, 1, 2, 3, 3],
        ),
        (
            ["A", "B"],
            [[0, 1, "foo", np.nan, 3, 4], ["a", "b", np.nan, np.nan, "d", "e"]],
            ["A", "B"],
            [0, 0, 1, 2, 3, 3],
        ),
    ],
)
def test_empty_lists(explode_subset, expected_dict, expected_index, expected_columns):
    # Tests handling of empty lists
    df = pd.DataFrame(
        {
            "A": [[0, 1], "foo", [], [3, 4]],
            "B": [["a", "b"], np.nan, [], ["d", "e"]],
        }
    )
    result = df.explode_wide(explode_subset)
    expected = pd.DataFrame(
        expected_dict, expected_index, expected_columns, dtype=object
    )
    tm.assert_frame_equal(result, expected)


def test_empty_lists_in_empty_lists():
    # Tests behavior when all lists in column to explode are empty
    df = pd.DataFrame({"A": [[], [], []], "B": [1, 2, 3]})
    result = df.explode_wide("A")

    expected = pd.DataFrame(
        [np.array([np.nan, np.nan, np.nan], dtype=object), [1, 2, 3]],
        index=["A", "B"],
        columns=[0, 1, 2],
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "explode_subset, expected_dict, expected_index, expected_columns",
    [
        (
            ["A"],
            [
                [0, 1, 2, "foo", np.nan, 3, 4],
                [1, 1, 1, 1, 1, 1, 1],
                [
                    ["a", "b", "c"],
                    ["a", "b", "c"],
                    ["a", "b", "c"],
                    np.nan,
                    [],
                    ["d", "e"],
                    ["d", "e"],
                ],
            ],
            ["A", "B", "C"],
            [0, 0, 0, 1, 2, 3, 3],
        ),
        (
            ["A", "C"],
            [
                [0, 1, 2, "foo", np.nan, 3, 4],
                [1, 1, 1, 1, 1, 1, 1],
                ["a", "b", "c", np.nan, np.nan, "d", "e"],
            ],
            ["A", "B", "C"],
            [0, 0, 0, 1, 2, 3, 3],
        ),
    ],
)
def test_multi_columns(explode_subset, expected_dict, expected_index, expected_columns):
    # Ensure behavior when exploding multiple columns,
    # handling mixed types, empty lists, and NaNs
    df = pd.DataFrame(
        {
            "A": [[0, 1, 2], "foo", [], [3, 4]],
            "B": 1,
            "C": [["a", "b", "c"], np.nan, [], ["d", "e"]],
        }
    )
    result = df.explode_wide(explode_subset)
    expected = pd.DataFrame(
        expected_dict, expected_index, expected_columns, dtype=object
    )
    tm.assert_frame_equal(result, expected)


def test_multi_index_rows():
    # Ensure explode_wide preserves the structure and labeling of MultiIndex on rows
    index = pd.MultiIndex.from_tuples([("a", 1), ("a", 2)])
    df = pd.DataFrame(
        {"A": [[1, 2], [3, 4]], "B": [["a", "b"], ["c", "d"]]}, index=index
    )
    result = df.explode_wide(["A", "B"])

    expected_columns = pd.MultiIndex.from_tuples(
        [("a", 1), ("a", 1), ("a", 2), ("a", 2)], names=[None, None]
    )
    expected = pd.DataFrame(
        [[1, 2, 3, 4], ["a", "b", "c", "d"]], index=["A", "B"], columns=expected_columns
    )
    tm.assert_frame_equal(result, expected)


def test_multi_index_columns():
    # Ensure explode_wide supports exploding data when columns are MultiIndex
    columns = pd.MultiIndex.from_tuples([("A", 1), ("B", 2)])
    df = pd.DataFrame(
        {("A", 1): [[1, 2], [3, 4]], ("B", 2): [["a", "b"], ["c", "d"]]},
        columns=columns,
    )
    result = df.explode_wide([("A", 1), ("B", 2)])

    expected_index = pd.MultiIndex.from_tuples([("A", 1), ("B", 2)])
    expected = pd.DataFrame(
        [[1, 2, 3, 4], ["a", "b", "c", "d"]], index=expected_index, columns=[0, 0, 1, 1]
    )
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "dict, index, expected_dict, expected_columns, expected_index",
    [
        (
            {"col1": [[1, 2], [3, 4]], "col2": ["foo", "bar"]},
            [0, 0],
            [[1, 2, 3, 4], ["foo", "foo", "bar", "bar"]],
            [0, 0, 0, 0],
            ["col1", "col2"],
        ),
        (
            {"col1": [[5, 6], [3, 4]], "col2": ["foo", "bar"]},
            pd.Index([0, 0], name="my_index"),
            [[5, 6, 3, 4], ["foo", "foo", "bar", "bar"]],
            pd.Index([0, 0, 0, 0], name="my_index"),
            ["col1", "col2"],
        ),
        (
            {"col1": [[1, 2], [3, 4]], "col2": ["foo", "bar"]},
            pd.MultiIndex.from_arrays(
                [[0, 0], [1, 1]], names=["my_first_index", "my_second_index"]
            ),
            [[1, 2, 3, 4], ["foo", "foo", "bar", "bar"]],
            pd.MultiIndex.from_arrays(
                [[0, 0, 0, 0], [1, 1, 1, 1]],
                names=["my_first_index", "my_second_index"],
            ),
            ["col1", "col2"],
        ),
        (
            {"col1": [[1, 2], [3, 4]], "col2": ["foo", "bar"]},
            pd.MultiIndex.from_arrays([[0, 0], [1, 1]], names=["my_index", None]),
            [[1, 2, 3, 4], ["foo", "foo", "bar", "bar"]],
            pd.MultiIndex.from_arrays(
                [[0, 0, 0, 0], [1, 1, 1, 1]], names=["my_index", None]
            ),
            ["col1", "col2"],
        ),
    ],
)
def test_duplicate_index(dict, index, expected_dict, expected_columns, expected_index):
    # Tests explode_wide with repeated and named row index values
    # Ensure duplication across different index types

    df = pd.DataFrame(dict, index=index, dtype=object)
    result = df.explode_wide("col1")
    expected = pd.DataFrame(
        expected_dict, columns=expected_columns, index=expected_index, dtype=object
    )
    tm.assert_frame_equal(result, expected)


def test_ignore_index():
    # Ensure reset of column labels to a simple range when ignore_index=True
    df = pd.DataFrame({"A": [[1, 2], [3, 4]], "B": [["a", "b"], ["c", "d"]]})
    result = df.explode_wide(["A", "B"], ignore_index=True)

    expected = pd.DataFrame(
        [[1, 2, 3, 4], ["a", "b", "c", "d"]], index=["A", "B"], columns=[0, 1, 2, 3]
    )
    tm.assert_frame_equal(result, expected)


def test_ignore_index_with_multi_index():
    # Ensure reset of column labels works with MultiIndex row index
    row_index = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)])
    df = pd.DataFrame(
        {"A": [[1, 2], [3, 4]]},
        index=row_index,
    )
    result = df.explode_wide("A", ignore_index=True)

    expected = pd.DataFrame([[1, 2, 3, 4]], index=["A"], columns=[0, 1, 2, 3])
    tm.assert_frame_equal(result, expected)


def test_ignore_index_with_duplicate_index():
    # Ensure reset of column labels works correctly with duplicate indices
    df = pd.DataFrame({"A": [[1, 2], [3, 4]], "B": ["x", "y"]}, index=[1, 1])
    result = df.explode_wide("A", ignore_index=True)

    expected = pd.DataFrame(
        [
            [1, 2, 3, 4],
            ["x", "x", "y", "y"],
        ],
        index=["A", "B"],
        columns=[0, 1, 2, 3],
    )
    tm.assert_frame_equal(result, expected)


def test_explode_wide_sets():
    # Ensure set values are sorted and expanded consistently across rows
    df = pd.DataFrame({"A": [{"z", "y"}, {"x"}], "B": [{"foo", "aab"}, {"bbc"}]})
    result = df.explode_wide(["A", "B"])

    expected = pd.DataFrame(
        [["y", "z", "x"], ["aab", "foo", "bbc"]], index=["A", "B"], columns=[0, 0, 1]
    )
    tm.assert_frame_equal(result, expected)


def test_preserve_column_order():
    # Ensure original column order is preserved when explosion order differs
    df = pd.DataFrame({"A": [[1, 2]], "B": [["x", "y"]], "C": [3]})
    result = df.explode_wide(["B", "A"])

    expected = pd.DataFrame(
        [[1, 2], ["x", "y"], [3, 3]],
        index=["A", "B", "C"],
        columns=[0, 0],
    )
    tm.assert_frame_equal(result, expected)
    assert list(result.index) == list(df.columns)
