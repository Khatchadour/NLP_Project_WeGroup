# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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


import csv
import json
import os

import datasets


_DESCRIPTION = """\
This dataset is used to finetune Large Language models from BLOOM family 
for Insurance policy data extraction from documents.
"""


_DATA_DIR  = "./data_finetuning"


class BloomInsurance(datasets.GeneratorBasedBuilder):
    """ Dataset of example query document in raw string format and 
    extracted data in a suitable format as an answer.
    """

    VERSION = datasets.Version("1.0")

    def _info(self):
        features = datasets.Features(
            {
                "question": datasets.Value("string"),
                "answer": datasets.Value("string")
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None
        )

    def _split_generators(self):
        data_dir = _DATA_DIR
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.txt"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "val.txt"),
                    "split": "val",
                },
            ),
        ]

    def _generate_examples(self, filepath, split):
        key = 0
        with open(filepath, encoding="utf-8") as f:   
            data = json.load(f)
            for example in data:
                yield key, {
                    "question": example["question"],
                    "answer": "" if split == "test" else example["answer"],
                }
                key += 1