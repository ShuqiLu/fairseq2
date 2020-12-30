# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('mask_decode_lm_para')
class MaskDecodeLmParaLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, task, tpu=False):
        super().__init__(task)
        self.tpu = tpu


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        masked_tokens = sample['target'].ne(self.padding_idx)
        sample_size_mask = masked_tokens.int().sum()

        decode_tokens=sample['decode_target'].ne(self.padding_idx)
        sample_size_decode=decode_tokens.int().sum()

        # Rare: when all tokens are masked, project all tokens.
        # We use torch.where to avoid device-to-host transfers,
        # except on CPU where torch.where is not well supported
        # (see github.com/pytorch/pytorch/issues/26247).
        if self.tpu:
            masked_tokens = None  # always project all tokens on TPU
        elif masked_tokens.device == torch.device('cpu'):
            if not masked_tokens.any():
                masked_tokens = None
            if not decode_tokens.any():
                decode_tokens=None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )
            decode_tokens = torch.where(
                decode_tokens.any(),
                decode_tokens,
                decode_tokens.new([True]),
            )

        logits, logits_decode, _ = model(**sample['net_input'], masked_tokens=masked_tokens, )
        targets = model.get_targets(sample, [logits])
        if masked_tokens is not None:
            targets = targets[masked_tokens]

        #print('???',logits_decode.shape)
        decode_target=sample["decode_target"]
        if decode_tokens is not None:
            if logits_decode.shape[1]!=decode_target.shape[1]:
                print(decode_target)
                print(sample['net_input']['src_tokens'])
            decode_target=decode_target[decode_tokens]
            logits_decode=logits_decode[decode_tokens]


        mask_loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )

        decode_loss = modules.cross_entropy(
            logits_decode.view(-1, logits_decode.size(-1)),
            decode_target.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )
        
        accumulate_step = sample['accumulate_step']

        logging_output = {
            #'loss': loss if self.tpu else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            #'sample_size': sample_size,
            'loss_decode' : decode_loss if self.tpu else decode_loss.data,
            'loss_mask' : mask_loss if self.tpu else mask_loss.data,
            'sample_size_decode':sample_size_decode ,
            'sample_size_mask': sample_size_mask,
            'sample_size': sample_size_mask,
            'sample_size_t': 1.0/accumulate_step,
            'loss' : mask_loss if self.tpu else mask_loss.data,
        }

        sample_size_mask = sample['sample_size_mask']
        sample_size_decode = sample['sample_size_decode']
        

        decode_loss=decode_loss/sample_size_decode
        mask_loss=mask_loss/sample_size_mask
        loss=0.5*mask_loss+0.5*decode_loss

        #print('???',decode_loss,mask_loss)

        return loss, 1.0/accumulate_step, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        # loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        #sample_size = sum(log.get('sample_size', 0) for log in logging_ouRtputs)
        #metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        

        
        loss_mask = sum(log.get('loss_mask', 0) for log in logging_outputs)
        loss_decode = sum(log.get('loss_decode', 0) for log in logging_outputs)
        sample_size_decode = sum(log.get('sample_size_decode', 0) for log in logging_outputs)
        sample_size_mask  = sum(log.get('sample_size_mask', 0) for log in logging_outputs)
        accumulate_step = sum(log.get('sample_size_t', 0) for log in logging_outputs)

        loss_decode=loss_decode / sample_size_decode / math.log(2)
        loss_mask=loss_mask / sample_size_mask / math.log(2)

        metrics.log_scalar('loss_decode', loss_decode, sample_size_decode, round=3)
        metrics.log_scalar('loss_mask', loss_mask, sample_size_mask, round=3)
        loss_sum=0.5*loss_mask+0.5*loss_decode
        metrics.log_scalar('loss', loss_sum , 0.5*sample_size_decode+0.5*sample_size_mask, round=3)


        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

        metrics.log_scalar('sample_size_t', accumulate_step , accumulate_step, round=3)

        token = sum(log.get('ntokens', 0) for log in logging_outputs)
        metrics.log_scalar('ntokens', token , token, round=3)
        metrics.log_scalar('sample_size_decode', sample_size_decode , sample_size_decode, round=3)

        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        metrics.log_scalar('sample_size', sample_size , sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
