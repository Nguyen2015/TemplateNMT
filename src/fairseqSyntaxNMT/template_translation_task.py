# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


import itertools
import logging
import os

from fairseq import utils
from fairseq.data import (
    data_utils, Dictionary, ConcatDataset,
    IndexedRawTextDataset, IndexedCachedDataset, IndexedDataset
)
from fairseq.data.language_pair_dataset import LanguagePairDataset
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

from . import TemplateLanguagePairDataset

logger = logging.getLogger(__name__)

    
@register_task('template_translation')
class TemplateTranslationTask(TranslationTask): 

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser.""" 
        TranslationTask.add_args(parser)
        parser.add_argument('--template-type', type=str,
                            default='probt',
                            help='template type')        
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')

    @staticmethod
    def load_pretrained_model(path, src_dict_path, tgt_dict_path, template_dict_path, arg_overrides=None):
        model = utils.load_checkpoint_to_cpu(path)
        args = model['args']
        state_dict = model['model']
        args = utils.override_model_args(args, arg_overrides)
        src_dict = Dictionary.load(src_dict_path)
        tgt_dict = Dictionary.load(tgt_dict_path)
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        template_dict = Dictionary.load(template_dict_path)

        task = TemplateTranslationTask(args, src_dict, tgt_dict, template_dict)
        model = task.build_model(args)
        model.upgrade_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=True)
        return model

    def __init__(self, args, src_dict, tgt_dict, template_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.args = args
        self.template_dict = template_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        translation_task = TranslationTask.setup_task(args, **kwargs)

        if args.template_type is None:
            raise Exception('Type of template data is missing')
        template_dict = Dictionary.load(os.path.join(args.data, 'dict.{}.txt'.format(args.template_type)))
        logger.info("[{}] dictionary: {} types".format(args.template_type, len(template_dict)))

        return cls(args, translation_task.src_dict, translation_task.tgt_dict, template_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary): 
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedDataset.exists(path):
                return IndexedCachedDataset(path, fix_lua_indexing=True)
            return None

        src_datasets = []
        tgt_datasets = []
        template_datasets = []

        data_paths = [self.args.data]

        for dk, data_path in enumerate(data_paths):
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')

                # infer langcode
                src, tgt = self.args.source_lang, self.args.target_lang
                if split_exists(split_k, src, tgt, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                elif split_exists(split_k, tgt, src, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                else:
                    if k > 0 or dk > 0:
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))
  
                # infer template data
                template_type = self.args.template_type 
                if split_exists(split_k, template_type, tgt, template_type, data_path):
                    prefix_template = os.path.join(data_path, '{}.{}-{}.'.format(split_k, template_type, tgt))
                else:
                    raise FileNotFoundError('Template dataset not found: {} ({})'.format(split, data_path))

                src_datasets.append(data_utils.load_indexed_dataset(prefix + src, self.src_dict, self.args.dataset_impl))
                tgt_datasets.append(data_utils.load_indexed_dataset(prefix + tgt, self.tgt_dict, self.args.dataset_impl))
                template_datasets.append(data_utils.load_indexed_dataset(prefix_template + template_type, self.template_dict, self.args.dataset_impl))
                logger.info('translation src-tgt {} {} {} examples'.format(data_path, split_k, len(src_datasets[-1])))
                logger.info('translation template-tgt {} {} {} examples'.format(data_path, split_k, len(template_datasets[-1])))

                if not combine:
                    break

        assert len(src_datasets) == len(tgt_datasets) == len(template_datasets)

        if len(src_datasets) == 1:
            src_dataset, tgt_dataset, template_dataset = src_datasets[0], tgt_datasets[0], template_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
            template_dataset = ConcatDataset(template_datasets, sample_ratios)

        self.datasets[split] = TemplateLanguagePairDataset(
            src_dataset, src_dataset.sizes, self.src_dict,
            tgt_dataset, tgt_dataset.sizes if tgt_dataset is not None else None, self.tgt_dict,
            template_dataset, template_dataset.sizes, self.template_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, template_tokens, template_tokens_sizes, constraints=None):
        return TemplateLanguagePairDataset(
            src_tokens, src_lengths, self.src_dict,
            tgt_dict=self.tgt_dict,
            template_tokens=template_tokens, 
            template_tokens_sizes=template_tokens_sizes, 
            template_tokens_dict=self.template_dict,
            constraints=constraints
        )

        # return LanguagePairDataset(
        #     src_tokens,
        #     src_lengths,
        #     self.source_dictionary,
        #     tgt_dict=self.target_dictionary,
        #     constraints=constraints,
        # )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    @property
    def template_dictionary(self):
        """Return the template dictionary :class:`~fairseq.data.Dictionary`."""
        return self.template_dict

