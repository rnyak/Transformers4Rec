#
# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest

from merlin_standard_lib import Tag
from tests.tf import _utils as test_utils

tf4rec = pytest.importorskip("transformers4rec.tf")


def test_sequence_embedding_features(yoochoose_schema, tf_yoochoose_like):
    schema = yoochoose_schema.select_by_tag(Tag.CATEGORICAL)
    emb_module = tf4rec.SequenceEmbeddingFeatures.from_schema(schema)

    outputs = emb_module(tf_yoochoose_like)

    assert list(outputs.keys()) == schema.select_by_tag(Tag.CATEGORICAL).column_names
    assert all(len(tensor.shape) == 3 for tensor in list(outputs.values()))
    assert all(tensor.shape[1] == 20 for tensor in list(outputs.values()))
    assert all(tensor.shape[2] == 64 for tensor in list(outputs.values()))


def test_serialization_sequence_embedding_features(yoochoose_schema, tf_yoochoose_like):
    inputs = tf4rec.SequenceEmbeddingFeatures.from_schema(yoochoose_schema)

    copy_layer = test_utils.assert_serialization(inputs)

    assert list(inputs.feature_config.keys()) == list(copy_layer.feature_config.keys())

    from transformers4rec.tf.features.embedding import serialize_table_config as ser

    assert all(
        ser(inputs.feature_config[key].table) == ser(copy_layer.feature_config[key].table)
        for key in copy_layer.feature_config
    )


@test_utils.mark_run_eagerly_modes
def test_sequence_embedding_features_yoochoose_model(
    yoochoose_schema, tf_yoochoose_like, run_eagerly
):
    inputs = tf4rec.TabularSequenceFeatures.from_schema(
        yoochoose_schema, max_sequence_length=20, aggregation="concat"
    )

    body = tf4rec.SequentialBlock([inputs, tf4rec.MLPBlock([64])])

    test_utils.assert_body_works_in_model(tf_yoochoose_like, inputs, body, run_eagerly)


def test_sequence_tabular_features_with_projection(yoochoose_schema, tf_yoochoose_like):
    schema = yoochoose_schema
    tab_module = tf4rec.TabularSequenceFeatures.from_schema(
        yoochoose_schema, max_sequence_length=20, continuous_projection=64
    )

    continuous_feature_names = schema.select_by_tag(Tag.CONTINUOUS).column_names

    outputs = tab_module(tf_yoochoose_like)

    assert len(set(continuous_feature_names).intersection(set(outputs.keys()))) == 0
    assert "continuous_projection" in outputs
    assert list(outputs["continuous_projection"].shape)[1:] == [20, 64]


def test_serialization_sequence_tabular_features(yoochoose_schema, tf_yoochoose_like):
    inputs = tf4rec.TabularSequenceFeatures.from_schema(yoochoose_schema)

    copy_layer = test_utils.assert_serialization(inputs)

    assert list(inputs.to_merge.keys()) == list(copy_layer.to_merge.keys())


@test_utils.mark_run_eagerly_modes
def test_tabular_features_yoochoose_direct(
    yoochoose_schema,
    tf_yoochoose_like,
    run_eagerly,
):
    continuous_layer = tf4rec.ContinuousFeatures.from_schema(yoochoose_schema, tags=["continuous"])
    categorical_layer = tf4rec.SequenceEmbeddingFeatures.from_schema(
        yoochoose_schema, tags=["categorical"]
    )

    tab_seq_features = tf4rec.TabularSequenceFeatures(
        continuous_layer=continuous_layer,
        categorical_layer=categorical_layer,
        aggregation="sequential-concat",
    )
    outputs = tab_seq_features(tf_yoochoose_like)

    assert (
        len(
            set(categorical_layer.schema.column_names).difference(
                set(tab_seq_features.schema.column_names)
            )
        )
        == 0
    )
    assert (
        len(
            set(continuous_layer.schema.column_names).difference(
                set(tab_seq_features.schema.column_names)
            )
        )
        == 0
    )
    assert len(outputs.shape) == 3


# Add these tests when we port Masking to TF

# def test_sequential_tabular_features_with_masking(yoochoose_schema, tf_yoochoose_like):
#     input_module = tf4rec.SequentialTabularFeatures.from_schema(
#         yoochoose_schema,
#         max_sequence_length=20,
#         continuous_projection=64,
#         d_output=100,
#         masking="causal",
#     )
#
#     outputs = input_module(tf_yoochoose_like)
#
#     assert outputs.ndim == 3
#     assert outputs.shape[-1] == 100
#     assert outputs.shape[1] == 20
#
#
# def test_sequential_tabular_features_with_masking_no_itemid(yoochoose_schema):
#     with pytest.raises(ValueError) as excinfo:
#
#         yoochoose_schema = yoochoose_schema - ["item_id/list"]
#
#         tf4rec.SequentialTabularFeatures.from_schema(
#             yoochoose_schema,
#             max_sequence_length=20,
#             continuous_projection=64,
#             d_output=100,
#             masking="causal",
#         )
#
#     err = excinfo.value
#     assert "For masking a categorical_module is required including an item_id" in str(err)


def test_sequence_tabular_features_with_projection_and_d_output(yoochoose_schema):
    with pytest.raises(ValueError) as excinfo:
        tf4rec.TabularSequenceFeatures.from_schema(
            yoochoose_schema,
            max_sequence_length=20,
            continuous_projection=64,
            d_output=100,
            projection=tf4rec.MLPBlock([64]),
            masking="causal",
        )

    assert "You cannot specify both d_output and projection at the same time" in str(excinfo.value)


def test_sequential_and_non_sequential_tabular_features(yoochoose_schema, tf_yoochoose_like):
    schema = yoochoose_schema
    tab_module = tf4rec.TabularSequenceFeatures.from_schema(schema, aggregation="concat")

    outputs = tab_module(tf_yoochoose_like)

    assert list(outputs.shape) == [100, 20, 199]
