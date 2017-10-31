import pytest
from ..batchmaker import *

PAD = 99

SORTED_DATA = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [0, 1, 2, 3, 4, 5, 6],
    [0, 1, 2, 3, 4, 5],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3],
    [0, 1, 2],
    [0, 1],
    [0]
]

SCRAMBLED_DATA = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [0, 1, 2],
    [0, 1],
    [0, 1, 2, 3, 4],
    [0, 1, 2, 3],
    [0],
    [0, 1, 2, 3, 4, 5, 6],
    [0, 1, 2, 3, 4, 5]
]

BATCHED_SORTED_DATA_2 = [
    [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6]
    ],
    [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 2, 3, 4]
    ],
    [
        [0, 1, 2, 3],
        [0, 1, 2]
    ],
    [
        [0, 1],
        [0]
    ]
]

BATCHED_SORTED_DATA_3 = [
    [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2, 3, 4, 5, 6],
        [0, 1, 2, 3, 4, 5]
    ],
    [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3],
        [0, 1, 2]
    ],
    [
        [0, 1],
        [0],
        []
    ]
]

BATCHED_SCRAMBLED_DATA_2 = [
    [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2]
    ],
    [
        [0, 1],
        [0, 1, 2, 3, 4]
    ],
    [
        [0, 1, 2, 3],
        [0]
    ],
    [
        [0, 1, 2, 3, 4, 5, 6],
        [0, 1, 2, 3, 4, 5]
    ]
]

BATCHED_SCRAMBLED_DATA_3 = [
    [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 2],
        [0, 1]
    ],
    [
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3],
        [0]
    ],
    [
        [0, 1, 2, 3, 4, 5, 6],
        [0, 1, 2, 3, 4, 5],
        []
    ]
]

TRUNCATED_SCRAMBLED_BATCHES_4 = [
    [
        [0, 1, 2, 3],
        [0, 1, 2]
    ],
    [
        [4, 5, 6, 7],
        []
    ],
    [
        [0, 1],
        [0, 1, 2, 3]
    ],
    [
        [],
        [4]
    ],
    [
        [0, 1, 2, 3],
        [0]
    ],
    [
        [0, 1, 2, 3],
        [0, 1, 2, 3]
    ],
    [
        [4, 5, 6],
        [4, 5]
    ]
]

TRUNCATED_SORTED_BATCHES_4 = [
    [
        [0, 1, 2, 3],
        [0, 1, 2, 3]
    ],
    [
        [4, 5, 6, 7],
        [4, 5, 6]
    ],
    [
        [0, 1, 2, 3],
        [0, 1, 2, 3]
    ],
    [
        [4, 5],
        [4]
    ],
    [
        [0, 1, 2, 3],
        [0, 1, 2]
    ],
    [
        [0, 1],
        [0]
    ]
]

TRUNCATED_SORTED_BATCHES_3_4 = [
    [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3]
    ],
    [
        [4, 5, 6, 7],
        [4, 5, 6],
        [4, 5]
    ],
    [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2]
    ],
    [
        [4],
        [],
        []
    ],
    [
        [0, 1],
        [0],
        []
    ]
]

SIZES_TRUNCATED_SCRAMBLED_BATCHES_4 = [
    [4, 3], [4, 0], [2, 4], [0, 1], [4, 1], [4, 4], [3, 2]
]

SIZES_TRUNCATED_SORTED_BATCHES_4 = [
    [4, 4], [4, 3], [4, 4], [2, 1], [4, 3], [2, 1]
]

SIZES_TRUNCATED_SORTED_BATCHES_3_4 = [
    [4, 4, 4], [4, 3, 2], [4, 4, 3], [1, 0, 0], [2, 1, 0]
]

PADDED_SCRAMBLED_BATCHES_4 = [
    [
        [0, 1, 2, 3],
        [0, 1, 2, PAD]
    ],
    [
        [4, 5, 6, 7],
        [PAD, PAD, PAD, PAD]
    ],
    [
        [0, 1, PAD, PAD],
        [0, 1, 2, 3]
    ],
    [
        [PAD, PAD, PAD, PAD],
        [4, PAD, PAD, PAD]
    ],
    [
        [0, 1, 2, 3],
        [0, PAD, PAD, PAD]
    ],
    [
        [0, 1, 2, 3],
        [0, 1, 2, 3]
    ],
    [
        [4, 5, 6, PAD],
        [4, 5, PAD, PAD]
    ]
]

PADDED_SORTED_BATCHES_4 = [
    [
        [0, 1, 2, 3],
        [0, 1, 2, 3]
    ],
    [
        [4, 5, 6, 7],
        [4, 5, 6, PAD]
    ],
    [
        [0, 1, 2, 3],
        [0, 1, 2, 3]
    ],
    [
        [4, 5, PAD, PAD],
        [4, PAD, PAD, PAD]
    ],
    [
        [0, 1, 2, 3],
        [0, 1, 2, PAD]
    ],
    [
        [0, 1, PAD, PAD],
        [0, PAD, PAD, PAD]
    ]
]

PADDED_SORTED_BATCHES_3_4 = [
    [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3]
    ],
    [
        [4, 5, 6, 7],
        [4, 5, 6, PAD],
        [4, 5, PAD, PAD]
    ],
    [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, PAD]
    ],
    [
        [4, PAD, PAD, PAD],
        [PAD, PAD, PAD, PAD],
        [PAD, PAD, PAD, PAD]
    ],
    [
        [0, 1, PAD, PAD],
        [0, PAD, PAD, PAD],
        [PAD, PAD, PAD, PAD]
    ]
]

class TestSortByLength():
    def test_should_not_change_data_when_it_is_empty(self):
        assert sort_by_length([]) == []

    def test_should_not_change_data_when_it_is_already_sorted(self):
        assert sort_by_length([SORTED_DATA]) == [SORTED_DATA]

    def test_should_sort_scrambled_data(self):
        assert sort_by_length([SCRAMBLED_DATA]) == [SORTED_DATA]

    def test_should_sort_all_data(self):
        assert sort_by_length([SORTED_DATA, SCRAMBLED_DATA, SCRAMBLED_DATA, SORTED_DATA]) == [
            SORTED_DATA, SORTED_DATA, SORTED_DATA, SORTED_DATA]

class TestGroupIntoBatches():
    def test_should_return_empty_list_when_data_is_empty(self):
        assert group_into_batches([], 4) == []

    def test_should_raise_value_error_when_batch_size_is_less_than_one(self):
        with pytest.raises(ValueError):
            group_into_batches([], 0)

    def test_should_not_raise_value_error_when_batch_size_is_one(self):
        group_into_batches([], 1)
        pass

    def test_should_create_batches_out_of_one_set_of_data(self):
        assert group_into_batches([SCRAMBLED_DATA], 2) == [BATCHED_SCRAMBLED_DATA_2]
        assert group_into_batches([SCRAMBLED_DATA], 3) == [BATCHED_SCRAMBLED_DATA_3]

    def test_should_create_batches_out_of_multiple_sets_of_data(self):
        assert group_into_batches([SCRAMBLED_DATA, SORTED_DATA], 2) == [BATCHED_SCRAMBLED_DATA_2, BATCHED_SORTED_DATA_2]

class TestTruncateBatches():
    def test_should_do_nothing_when_data_is_empty(self):
        assert truncate_batches([], 2) == []

    def test_should_raise_value_error_when_truncate_is_less_than_one(self):
        with pytest.raises(ValueError):
            truncate_batches([], 0)

    def test_should_not_raise_value_error_when_truncate_is_one(self):
        truncate_batches([], 1)
        pass

    def test_should_return_same_data_when_truncate_is_equal_to_longest_example(self):
        assert truncate_batches(BATCHED_SCRAMBLED_DATA_2, 8) == BATCHED_SCRAMBLED_DATA_2

    def test_should_return_batches_with_length_less_than_or_equal_to_truncate(self):
        for i in range(1, 10):
            truncated_data = truncate_batches(BATCHED_SCRAMBLED_DATA_2, i)
            for batch in truncated_data:
                for example in batch:
                    assert len(example) <= i

    def test_should_correctly_truncate_data(self):
        assert truncate_batches(BATCHED_SCRAMBLED_DATA_2, 4) == TRUNCATED_SCRAMBLED_BATCHES_4
        assert truncate_batches(BATCHED_SORTED_DATA_3, 4) == TRUNCATED_SORTED_BATCHES_3_4
        assert truncate_batches(BATCHED_SORTED_DATA_2, 4) == TRUNCATED_SORTED_BATCHES_4

class TestGetRowLenghts():
    def test_should_return_empty_list_when_data_is_empty(self):
        assert get_row_lengths([]) == []

    def test_should_return_row_lengths_for_one_piece_of_data(self):
        assert get_row_lengths(TRUNCATED_SORTED_BATCHES_4) == SIZES_TRUNCATED_SORTED_BATCHES_4
        assert get_row_lengths(TRUNCATED_SCRAMBLED_BATCHES_4) == SIZES_TRUNCATED_SCRAMBLED_BATCHES_4
        assert get_row_lengths(TRUNCATED_SORTED_BATCHES_3_4) == SIZES_TRUNCATED_SORTED_BATCHES_3_4

class TestPadBatches():
    def test_should_return_empty_list_when_data_is_empty(self):
        assert pad_batches([], 2, PAD).tolist() == []

    def test_should_throw_value_error_when_truncate_is_less_than_one(self):
        with pytest.raises(ValueError):
            pad_batches([], 0, PAD)

    def test_should_not_throw_value_error_when_truncate_is_one(self):
        pad_batches([], 1, PAD)
        pass

    def test_should_correctly_pad_data(self):
        assert pad_batches(TRUNCATED_SCRAMBLED_BATCHES_4, 4, PAD).tolist() == PADDED_SCRAMBLED_BATCHES_4
        assert pad_batches(TRUNCATED_SORTED_BATCHES_4, 4, PAD).tolist() == PADDED_SORTED_BATCHES_4
        assert pad_batches(TRUNCATED_SORTED_BATCHES_3_4, 4, PAD).tolist() == PADDED_SORTED_BATCHES_3_4

class TestMakeBatches():
    def test_should_do_nothing_when_data_is_empty(self):
        x_batches, y_batches, lengths = make_batches([], [], 2, 4, PAD)
        assert x_batches.tolist() == []
        assert y_batches.tolist() == []
        assert lengths == []

    def test_should_raise_value_error_when_batch_size_is_less_than_one(self):
        with pytest.raises(ValueError):
            make_batches([], [], -1, 4, PAD)

    def test_should_not_raise_value_error_when_batch_size_is_one(self):
        make_batches([], [], 1, 4, PAD)
        pass

    def test_should_raise_value_error_when_truncate_is_less_than_one(self):
        with pytest.raises(ValueError):
            make_batches([], [], 2, -1, PAD)

    def test_should_not_raise_value_error_when_truncate_is_one(self):
        make_batches([], [], 2, 1, PAD)
        pass

    def test_should_correctly_create_batches(self):
        x_batches, y_batches, lengths = make_batches(SCRAMBLED_DATA, SORTED_DATA, 2, 4, PAD)
        for batch in x_batches:
            for example in batch:
                assert len(example) == 4
        for batch in y_batches:
            for example in batch:
                assert len(example) <= 4
        assert x_batches.tolist() == PADDED_SORTED_BATCHES_4
        assert y_batches.tolist() == PADDED_SORTED_BATCHES_4
        assert lengths == SIZES_TRUNCATED_SORTED_BATCHES_4
