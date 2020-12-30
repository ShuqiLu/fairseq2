# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import BaseWrapperDataset


class CheckEosDataset(BaseWrapperDataset):

    def __init__(self, dataset, id_to_replace, replace_id):
        super().__init__(dataset)
        self.id_to_replace = id_to_replace
        self.replace_id=replace_id

    def __getitem__(self, index):
        item = self.dataset[index]
        # while len(item) > 0 and item[-1] == self.id_to_strip:
        #     item = item[:-1]
        # while len(item) > 0 and item[0] == self.id_to_strip:
        #     item = item[1:]
        # return item
        if len(item)==1 and item[-1]==self.id_to_replace:
            item[-1]=self.replace_id
        return item
