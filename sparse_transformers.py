import torch_blocksparse as tbs
from torch_blocksparse import BertSparseSelfAttention, SparsityConfig
from transformers.modeling_bert import BertModel, BertConfig, BertForSequenceClassification
from transformers.modeling_roberta import RobertaConfig, RobertaForSequenceClassification 
import types
'''
This file contains few utility functions to handle adapting pretrained model with sparse self-attention module.
'''
class SparseBert(BertModel):
    def extend_position_embedding(self, max_position):
        original_max_position = self.embeddings.position_embeddings.weight.size(0)
        if max_position <= original_max_position:
            print(f'Not extending position embedding since {riginal_max_positioni} >= {max_position}')
            return

        extend_multiples = max(1, max_position // original_max_position)
        self.embeddings.position_embeddings.weight.data = self.embeddings.position_embeddings.weight.repeat(
            extend_multiples,
            1)

        self.config.max_position_embeddings = max_position
        self.embeddings.position_embeddings.num_embeddings = max_position
        print(f'Extended position embeddings to {original_max_position * extend_multiples}')


    def replace_model_self_attention_with_sparse_self_attention(
        self,
        max_position,
        sparsity_config=SparsityConfig(num_heads=4)):

        self.config.max_position_embeddings = max_position
        tbs.replace_self_attention_layer_with_sparse_self_attention_layer(
            self.config,
            self.encoder.layer,
            sparsity_config)


    def make_long_and_sparse(self, seq_len, sparsity, block_size, different_layout_per_head):
        if seq_len > self.config.max_position_embeddings:
            self.extend_position_embedding(seq_len)

        if sparsity == "fixed":
            self.sparsity_config = tbs.FixedSparsityConfig(
                    num_heads = self.config.num_attention_heads,
                    seq_len = seq_len,
                    block = block_size,
                    different_layout_per_head = different_layout_per_head,
                    attention = 'bidirectional')
        elif sparsity == "bigbird": 
            self.sparsity_config = tbs.BigBirdSparsityConfig(
                    num_heads = self.config.num_attention_heads,
                    seq_len = seq_len,
                    block = block_size,
                    num_sliding_window_blocks=3,
                    num_global_blocks=1,
                    different_layout_per_head = different_layout_per_head)

        elif sparsity == "longformer":
            self.sparsity_config = tbs.BSLongformerSparsityConfig(
                    num_heads = self.config.num_attention_heads,
                    seq_len = seq_len,
                    block = block_size,
                    num_sliding_window_blocks=3,
                    global_block_indices=[0],
                    different_layout_per_head = different_layout_per_head)
        else: 
            raise NotImplementedError

        if self.sparsity_config:
            self.replace_model_self_attention_with_sparse_self_attention(
                    seq_len, self.sparsity_config)


# class SparseRoBerta(RobertaModel):
#     def extend_position_embedding(self, max_position):
#         original_max_position = self.embeddings.position_embeddings.weight.size(0)
#         if max_position <= original_max_position:
#             print(f'Not extending position embedding since {riginal_max_positioni} >= {max_position}')
#             return

#         extend_multiples = max(1, max_position // original_max_position)
#         self.embeddings.position_embeddings.weight.data = self.embeddings.position_embeddings.weight.repeat(
#             extend_multiples,
#             1)

#         self.config.max_position_embeddings = max_position
#         self.embeddings.position_embeddings.num_embeddings = max_position
#         print(f'Extended position embeddings to {original_max_position * extend_multiples}')


#     def replace_model_self_attention_with_sparse_self_attention(
#         self,
#         max_position,
#         sparsity_config=SparsityConfig(num_heads=4,
#                                        seq_len=1024)):

#         self.config.max_position_embeddings = max_position
#         tbs.replace_self_attention_layer_with_sparse_self_attention_layer(
#             self.config,
#             self.encoder.layer,
#             sparsity_config)


#     def make_long_and_sparse(self, seq_len, sparsity, block_size, different_layout_per_head,num_sliding_window_blocks,global_block_indices):
#         if seq_len > self.config.max_position_embeddings:
#             self.extend_position_embedding(seq_len)

#         if sparsity == "fixed":
#             self.sparsity_config = tbs.FixedSparsityConfig(
#                     num_heads = self.config.num_attention_heads,
#                     seq_len = seq_len,
#                     block = block_size,
#                     different_layout_per_head = different_layout_per_head,
#                     attention = 'bidirectional')
#         elif sparsity == "bigbird": 
#             self.sparsity_config = tbs.BigBirdSparsityConfig(
#                     num_heads = self.config.num_attention_heads,
#                     seq_len = seq_len,
#                     block = block_size,
#                     num_sliding_window_blocks=3,
#                     num_global_blocks=1,
#                     different_layout_per_head = different_layout_per_head)

#         elif sparsity == "longformer":
#             self.sparsity_config = tbs.BSLongformerSparsityConfig(
#                     num_heads = self.roberta.config.num_attention_heads,
#                     seq_len = seq_len,
#                     block = block_size,
#                     # num_sliding_window_blocks=3,
#                     # global_block_indices=[0],
#                     num_sliding_window_blocks=num_sliding_window_blocks,
#                     global_block_indices=global_block_indices,
#                     different_layout_per_head = different_layout_per_head)
#         else: 
#             raise NotImplementedError

#         if self.sparsity_config:
#             self.replace_model_self_attention_with_sparse_self_attention(
#                     seq_len, self.sparsity_config)


class SparseBertForSequenceClassification(BertForSequenceClassification):
    def extend_position_embedding(self, max_position):
        original_max_position = self.bert.embeddings.position_embeddings.weight.size(0)
        if max_position <= original_max_position:
            print(f'Not extending position embedding since {riginal_max_positioni} >= {max_position}')
            return

        extend_multiples = max(1, max_position // original_max_position)
        self.bert.embeddings.position_embeddings.weight.data = self.bert.embeddings.position_embeddings.weight.repeat(
            extend_multiples,
            1)

        self.bert.config.max_position_embeddings = max_position
        self.bert.embeddings.position_embeddings.num_embeddings = max_position
        print(f'Extended position embeddings to {original_max_position * extend_multiples}')

    def make_long_and_sparse(self, seq_len, sparsity, block_size, different_layout_per_head):
        if seq_len > self.bert.config.max_position_embeddings:
            self.extend_position_embedding(seq_len)

        if sparsity == "fixed":
            self.sparsity_config = tbs.FixedSparsityConfig(
                    num_heads = self.bert.config.num_attention_heads,
                    seq_len = seq_len,
                    block = block_size,
                    different_layout_per_head = different_layout_per_head,
                    attention = 'bidirectional')
        elif sparsity == "bigbird": 
            self.sparsity_config = tbs.BigBirdSparsityConfig(
                    num_heads = self.bert.config.num_attention_heads,
                    seq_len = seq_len,
                    block = block_size,
                    num_sliding_window_blocks=3,
                    num_global_blocks=1,
                    different_layout_per_head = different_layout_per_head)

        elif sparsity == "longformer":
            self.sparsity_config = tbs.BSLongformerSparsityConfig(
                    num_heads = self.bert.config.num_attention_heads,
                    seq_len = seq_len,
                    block = block_size,
                    num_sliding_window_blocks=3,
                    global_block_indices=[0],
                    different_layout_per_head = different_layout_per_head)
        else: 
            raise NotImplementedError

        if self.sparsity_config:
            tbs.replace_model_self_attention_with_sparse_self_attention(
                    self, seq_len, self.sparsity_config)


class SparseRobertaForSequenceClassification(RobertaForSequenceClassification):
    def extend_position_embedding(self, max_position):

        original_max_position, embed_size = self.roberta.embeddings.position_embeddings.weight.shape
        original_max_position -= 2
        print('???',original_max_position)
        extend_multiples = max(1, max_position // original_max_position)
        assert max_position > original_max_position
        max_position += 2
        extended_position_embedding = self.roberta.embeddings.position_embeddings.weight.new_empty(
            max_position,
            embed_size)
        k = 2
        for i in range(extend_multiples):
            extended_position_embedding[k:(
                k + original_max_position
            )] = self.roberta.embeddings.position_embeddings.weight[2:]
            k += original_max_position
        self.roberta.embeddings.position_embeddings.weight.data = extended_position_embedding
        self.roberta.config.max_position_embeddings = max_position
        self.roberta.embeddings.position_embeddings.num_embeddings = max_position

    def make_long_and_sparse(self, seq_len, sparsity, block_size, different_layout_per_head,num_sliding_window_blocks,global_block_indices):
        if seq_len > self.roberta.config.max_position_embeddings:
            self.extend_position_embedding(seq_len)

        if sparsity == "fixed":
            self.sparsity_config = tbs.FixedSparsityConfig(
                    num_heads = self.roberta.config.num_attention_heads,
                    seq_len = seq_len,
                    block = block_size,
                    different_layout_per_head = different_layout_per_head,
                    attention = 'bidirectional')
        elif sparsity == "bigbird": 
            self.sparsity_config = tbs.BigBirdSparsityConfig(
                    num_heads = self.roberta.config.num_attention_heads,
                    seq_len = seq_len,
                    block = block_size,
                    num_sliding_window_blocks=3,
                    num_global_blocks=1,
                    different_layout_per_head = different_layout_per_head)

        elif sparsity == "longformer":
            self.sparsity_config = tbs.BSLongformerSparsityConfig(
                    num_heads = self.roberta.config.num_attention_heads,
                    #seq_len = seq_len,
                    block = block_size,
                    # num_sliding_window_blocks=3,
                    # global_block_indices=[0],
                    num_sliding_window_blocks=num_sliding_window_blocks,
                    global_block_indices=global_block_indices,
                    different_layout_per_head = different_layout_per_head)
        elif sparsity=="variable":
            self.sparsity_config = tbs.VariableSparsityConfig(
                    num_heads = self.roberta.config.num_attention_heads,
                    #seq_len = seq_len,
                    block = block_size,
                    # num_sliding_window_blocks=3,
                    # global_block_indices=[0],
                    local_window_blocks=num_sliding_window_blocks,
                    global_block_indices=global_block_indices,
                    different_layout_per_head = different_layout_per_head,
                    num_random_blocks=0,
                    attention='bidirectional',
                    horizontal_global_attention=True,
                    )
        else: 
            raise NotImplementedError

        if self.sparsity_config:
            tbs.replace_model_self_attention_with_sparse_self_attention(
                    self, seq_len, self.sparsity_config)

