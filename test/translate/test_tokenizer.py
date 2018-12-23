"""Tests the Tokenizer translator.
"""

import pytest
import string
import random

import numpy as np

from tf_rnn.translate import Tokenizer


NUM_INT_TOKENS = 30
INT_TOKENS = random.sample(range(100), NUM_INT_TOKENS)
INT_DATA = np.random.choice(INT_TOKENS, (100, 100)).tolist()

NUM_STR_TOKENS = 35
STR_TOKENS = random.sample(string.ascii_letters, NUM_STR_TOKENS)
STR_DATA = [random.choices(STR_TOKENS, k=100) for _ in range(100)]

class TestCreate():
    def test_should_create_tokenizer_with_correct_vocab_length(self):
        tokenizer = Tokenizer.create(INT_DATA)
        assert len(tokenizer.index_to_token) == NUM_INT_TOKENS
        assert len(tokenizer.token_to_index) == NUM_INT_TOKENS

        tokenizer = Tokenizer.create(STR_DATA)
        assert len(tokenizer.index_to_token) == NUM_STR_TOKENS
        assert len(tokenizer.token_to_index) == NUM_STR_TOKENS

class TestIndex():
    def test_should_convert_int_tokens_to_integers_with_the_correct_range(self):
        tokenizer = Tokenizer.create(INT_DATA)
        indexed = tokenizer.to_rnn_matrix(INT_TOKENS)
        assert np.sum(indexed >= NUM_INT_TOKENS) == 0
        assert np.sum(indexed < 0) == 0

    def test_should_convert_str_tokens_to_integers_with_the_correct_range(self):
        tokenizer = Tokenizer.create(STR_DATA)
        indexed = tokenizer.to_rnn_matrix(STR_TOKENS)
        assert np.sum(indexed >= NUM_STR_TOKENS) == 0
        assert np.sum(indexed < 0) == 0

class TestTokenize():
    def test_should_get_back_original_int_tokens(self):
        tokenizer = Tokenizer.create(INT_DATA)
        indexed = tokenizer.to_rnn_matrix(INT_DATA)
        tokenized = tokenizer.to_human_matrix(indexed)
        assert np.array_equal(tokenized, np.asarray(INT_DATA))

    def test_should_get_back_original_str_tokens(self):
        tokenizer = Tokenizer.create(STR_DATA)
        indexed = tokenizer.to_rnn_matrix(STR_DATA)
        tokenized = tokenizer.to_human_matrix(indexed)
        assert np.array_equal(tokenized, STR_DATA)
