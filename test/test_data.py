"""Data for testing.
"""

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
        True, [0, 1, 2, 3],
        [0, 1, 2], False
    ],
    [
        False, [4, 5, 6, 7],
        [], True
    ],
    [
        True, [0, 1],
        [0, 1, 2, 3], False
    ],
    [
        False, [],
        [4], True
    ],
    [
        True, [0, 1, 2, 3],
        [0], True
    ],
    [
        True, [0, 1, 2, 3],
        [0, 1, 2, 3], False
    ],
    [
        False, [4, 5, 6],
        [4, 5], True
    ]
]

TRUNCATED_SORTED_BATCHES_4 = [
    [
        True, [0, 1, 2, 3],
        [0, 1, 2, 3], False
    ],
    [
        False, [4, 5, 6, 7],
        [4, 5, 6], True
    ],
    [
        True, [0, 1, 2, 3],
        [0, 1, 2, 3], False
    ],
    [
        False, [4, 5],
        [4], True
    ],
    [
        True, [0, 1, 2, 3],
        [0, 1, 2], True
    ],
    [
        True, [0, 1],
        [0], True
    ]
]

TRUNCATED_SORTED_BATCHES_3_4 = [
    [
        True, [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3], False
    ],
    [
        False, [4, 5, 6, 7],
        [4, 5, 6],
        [4, 5], True
    ],
    [
        True, [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2], False
    ],
    [
        False, [4],
        [],
        [], True
    ],
    [
        True, [0, 1],
        [0],
        [], True
    ]
]

SIZES_TRUNCATED_SCRAMBLED_BATCHES_4 = [
    [True, 4, 3, False], [False, 4, 0, True], [True, 2, 4, False], [False, 0, 1, True], [True, 4, 1, True],
    [True, 4, 4, False], [False, 3, 2, True]
]

SIZES_TRUNCATED_SORTED_BATCHES_4 = [
    [True, 4, 4, False], [False, 4, 3, True], [True, 4, 4, False], [False, 2, 1, True], [True, 4, 3, True],
    [True, 2, 1, True]
]

SIZES_TRUNCATED_SORTED_BATCHES_3_4 = [
    [True, 4, 4, 4, False], [False, 4, 3, 2, True], [True, 4, 4, 3, False], [False, 1, 0, 0, True],
    [True, 2, 1, 0, True]
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

MAX_LENGTH = 8