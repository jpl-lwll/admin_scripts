# Copyright (c) 2023 California Institute of Technology (“Caltech”). U.S.
# Government sponsorship acknowledged.
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Caltech nor its operating division, the Jet Propulsion
#   Laboratory, nor the names of its contributors may be used to endorse or
#   promote products derived from this software without specific prior written
#   permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import boto3
import errno
import io
import numpy as np
import os
import pandas as pd
import sys
import json
from typing import Tuple

def create_label_budgets(*, fb_metadata: dict, task_type: str, num: int = 8) -> Tuple[list, list]:
    """
    Calculates the cumulative number of labels that can be retrieved at each checkpoint based on the total_size of the dataset.
    The first 4 checkpoints are set to enable few-shot learning and are [1n, 2n, 4n, 8n], where n is the number of
    classes. The number of labels retrieved at the remaining checkpoints is on a logarithmic scale from the number at the 4th checkpoint
    to the total size of the dataset.
    Args:
        name: fb_metadata. The FireBase metadata for the dataset represented as a dictionary. 
        num: int. The total number of checkpoints to generate
    Returns:
        checkpoints: list, list. In order sample, full. e.g.:
             [10, 10*2, ..., 6400], [10, 10*10, ..., 32000]

    """
    # calculate for sample and full
    size_types = ['sample', 'full']
    few_shot_chkpts = [1, 2, 4, 8]
    budgets_dict = {}  # store the label budgets
    for size_type in size_types:
        checkpoints = []
        if task_type in ['image_classification', 'object_detection', 'video_classification']:
            num_classes = fb_metadata[f'{size_type}_number_of_classes']
            checkpoints.extend([num_classes*i for i in few_shot_chkpts])
            num_add_chkpts = num - len(checkpoints)

            log_checkpoints = list(np.geomspace(checkpoints[-1], fb_metadata[f'{size_type}_number_of_samples_train'],  # noqa
                                                num_add_chkpts+1, endpoint=True))
            log_checkpoints = [int(round(i)) for i in log_checkpoints[1:]]
            checkpoints.extend(log_checkpoints)
        elif task_type == 'machine_translation':
            log_checkpoints = list(np.geomspace(5000, fb_metadata[f'{size_type}_total_codecs'],  # noqa
                                                num, endpoint=True))
            log_checkpoints = [int(round(i)) for i in log_checkpoints]
            checkpoints.extend(log_checkpoints)
        else:
            raise Exception(f'Invalid Problem Type. Got type {task_type}.')

        budgets_dict[size_type] = checkpoints
    return budgets_dict['sample'], budgets_dict['full']


def add_budget_labels_to_task(tasks: list, sample: bool = False, local_metadata_path=None) -> None:
    """
    Takes a list of task dictionaries and adds the label budgets for the base and adaptation sets, full and sample
    Args:
        tasks: list,
        [
            {
                "task_id": str,
                "problem_type": str,
                "base_dataset": str,
                "base_evaluation_metrics": list,
                "adaptation_dataset": str,
                "adaptation_evaluation_metrics": list,
                "base_can_use_pretrained_model": bool,
                "adaptation_can_use_pretrained_model": bool,
            },
            ...
            ]
        sample: bool, default False. If true creates budget labels for sample sets. Only dev tasks have samples.
    Return:
        None, list is modified in place
    """
    for task in tasks:
        dataset_types = ['base', 'adaptation']
        for dataset_type in dataset_types:
            dataset_name = task[f'{dataset_type}_dataset']
            if not local_metadata_path:
                # Import firebase store from lwll_api if not retrieving locally
                sys.path.insert(0, os.path.abspath('../lwll_api'))
                from lwll_api.classes.fb_auth import fb_store

                fb_metadata = fb_store.collection(
                    'DatasetMetadata').document(dataset_name).get().to_dict()
                if not fb_metadata:
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), dataset_name
                    )
            else:
                path = os.path.join(local_metadata_path, dataset_name, 'metadata.json')
                if not os.path.exists(path):
                    raise FileNotFoundError(
                        errno.ENOENT, os.strerror(errno.ENOENT), path
                    )
                
                with open(path, 'r') as metadata_reader:
                    fb_metadata = json.load(metadata_reader)

            task_type = task['problem_type']
            if task_type in ['image_classification', 'object_detection', 'video_classification']:
                sample_label_budget, full_label_budget = create_label_budgets(
                    fb_metadata=fb_metadata, task_type=task_type)
            else:
                sample_label_budget, full_label_budget = create_label_budgets(
                    fb_metadata=fb_metadata, task_type=task_type)
            if sample:
                task[f"{dataset_type}_label_budget_sample"] = sample_label_budget  # noqa
            task[f"{dataset_type}_label_budget_full"] = full_label_budget


class SeedLabelGenerator(object):

    def __init__(self, root_folder=None, profile_name="saml-pub"):
        self.root_folder = root_folder
        self.session = None
        self.s3 = None
        self.use_s3 = self.root_folder is None

        # If root_folder is not specified, default to using S3 with the specified profile
        if self.use_s3:
            self.session = boto3.Session(profile_name=profile_name)
            self.s3 = self.session.client("s3")

    def get_train_labels(self, dataset_name: str, level: str = 'dev', sample=False):
        """
        Gets the labels for the specified dataset.
        Args:
            dataset_name: str, the name of the dataset
            level: str, 'dev', 'staging', or 'eval'. This determines the s3 bucket to look in
            sample: bool, default False. Whether to get labels for the sample dataset or the full dataset.
        """
        s3_bucket = 'lwll-eval-datasets' if level == "eval" else 'lwll-datasets'
        size = 'sample' if sample else 'full'
        label_path = f"live/labels/{dataset_name}/{size}/labels_train.feather"

        # If root_folder is specified, use that to locally get the labels
        if not self.use_s3:
            label_path = os.path.join(self.root_folder, label_path)
            print(f"getting labels from {label_path}")
            return pd.read_feather(label_path)
        else:
            print(f"getting labels from {label_path} in bucket {s3_bucket}")
            obj = self.s3.get_object(Bucket=s3_bucket, Key=label_path)
            body = obj['Body'].read()
            df = pd.read_feather(io.BytesIO(body))
            return df

    def _check_seeds(self, lbls, lbls_per_class, problem_type):
        """
        Gets a dataframe of allocated seeds, the number of extra seeds needed and the number of classes to pull from
        Args:
            lbls: A dataframe of labels to pull from
            lbls_per_class: int, the number of labels to take per class
            problem_type: str, `image_classification`, `object_detection`, or `video_classification`
        Return
            (seed_allocation, num_classes_left)
        """
        class_size = lbls.groupby('class').size()
        num_classes_left = 0  # track how many classes will have extra labels in case one class runs out
        for cls in class_size.index:
            size = class_size[cls]
            if size > lbls_per_class:
                num_classes_left += 1
        seeds = lbls.groupby('class').apply(lambda x: x.head(lbls_per_class))
        if problem_type == "object_detection":
            # Make sure we get all of the labels for the selected ids since there can be multiple objects in one image
            seeds = self._get_all_lbls(lbls, seeds)
        elif problem_type == "video_classification":
            print(f"video labels: {seeds.shape[0]}")
            print(f"Getting all video labels")
            seeds = self._get_all_vid_labels(lbls, seeds['end_frame'].values)
            # Make sure that video_id and id are str and start and end frame are int
            seeds['video_id'] = seeds['video_id'].astype('str')
            seeds['id'] = seeds['id'].astype('str')
            seeds['start_frame'] = seeds['start_frame'].astype('int64')
            print(seeds.shape[0])
        return seeds, num_classes_left

    @staticmethod
    def _get_all_lbls(lbl_pool: pd.DataFrame, choices: pd.DataFrame):
        """
        Gets all the labels for the ids in lbl_choices. This is used for object detection where an image can have multiple
        objects in it.
        Args:
            lbl_pool: pd.DataFrame, A dataframe of all available labels
            choices: pd.DataFrame, A dataframe of selected labels
        """
        return lbl_pool.loc[lbl_pool['id'].isin(choices['id'])]

    @staticmethod
    def _drop_lbls(lbl_pool: pd.DataFrame, choices: pd.DataFrame):
        """
        Drops the selected seed ids from the pool that we sample from
        """
        if 'id' in lbl_pool.columns:
            return lbl_pool.loc[~lbl_pool['id'].isin(choices['id'])]

    @staticmethod
    def video_remove_multilabel(lbls: pd.DataFrame, num_total_lbls: int):
        """
        Some video frames belong to more than one segment. When we give labels for a frame, we want to give all the labels
        for a frame, but to avoid giving freebies, we want to drop segments with over lapping frames if possible.
        Args:
            lbls: A dataframe of video classification labels. Has form:
            segment_id  clip    start_frame   end_frame label
            1           1       0               10      a
            2           1       5               15      b
            3           1       25              35      a
            ...
            100         2       1000            1010    b
            101         2       1020            1030    c
            102         2       1025            1035    d
            num_total_lbls: int, the total number of seed labels to be given
        :return:
        """
        print(f"Checking video segments for multilabels. Num labels at start: {lbls.shape[0]}. Num examples needed is: {num_total_lbls}")
        if lbls.shape[0] <= num_total_lbls:
            print("Returning as is because there aren't extra labels")
            return lbls
        # First make sure the dataframe is sorted by start_frame
        lbls = lbls.sort_values(by="start_frame")
        to_drop = set()
        for i in range(lbls.shape[0]-2):
            this_row = lbls.iloc[i,:]
            next_row = lbls.iloc[i+1,:]
            next_next_row = lbls.iloc[i+2,:]
            # Check if the current segment overlaps with the next segment
            if next_row['start_frame'] >= this_row['start_frame'] and next_row['start_frame'] <= this_row['end_frame'] \
                    and lbls.shape[0] - 2 >= num_total_lbls:
                to_drop.add(this_row['id'])
                to_drop.add(next_row['id'])
            # Check if the current segment overlaps with 2 segments ahead
            if next_next_row['start_frame'] >= this_row['start_frame'] and next_next_row['start_frame'] <= this_row['end_frame'] \
                    and lbls.shape[0] - 2 >= num_total_lbls:
                to_drop.add(this_row['id'])
                to_drop.add(next_row['id'])
            # if lbls.shape[0] - len(to_drop) <= num_total_lbls:
            #     print(f"Breaking out of for loop bc to_drop is {len(to_drop)}")
            #     break
        print(f"Number of ids to drop is: {len(to_drop)}")
        lbls = lbls[~lbls.id.isin(to_drop)]
        print(f"Done checking video segments. Num labels is now: {lbls.shape[0]}")
        return lbls

    @staticmethod
    def _get_all_vid_labels(lbls: pd.DataFrame, end_frames: list):
        """
        In the case that a frame of any of the seeds segments also appears in another segment, we want to return all of
        the labels associated with the frame. In most instances this should not happen as `video_remove_multilabel` will
        have removed such cases. This will only happen in datasets that are too small to filter out all the segments that
        have a multilabel frame.
        Args:
            lbls: pd.DataFrame, pandas dataframe of all the train labels
            end_frames: list, a list of end_frame ids for seed labels
        :return:
        """
        labels = []
        for end_frame in end_frames:
            if isinstance(end_frame, str):
                end_frame = int(end_frame)
            label = lbls.loc[(end_frame >= lbls['start_frame']) & (end_frame <= lbls['end_frame'])]
            if label.shape[0] > 1:
                print(f"end_frame {end_frame} has overlapping segments:")
                print(label)
            labels.append(label)
        all_seeds = pd.concat(labels)
        return all_seeds

    def create_seed_lbls(self, lbls: pd.DataFrame, budgets_per_class=[1, 2, 4, 8], problem_type='image_classification'):
        """
        Creates a dictionary of the seed labels for the first four checkpoints. At each checkpoint, we start by
        getting the top n from each class. However, we have to handle the case where a class may be under represented
        and not have enough labels to meet the budget. In this case, we try to sample evenly from the other classes.
        Args:
            lbls: pd.DataFrame, A DataFrame containing all the labels for the dataset. Should be of
            the form
                | | class  | id       |
                |0| class1 | img1.png |
                |i| classk | imgj.png |
                |.|  ...   |    ...   |
            budgets_per_class: list of ints, the number of cumulative labels given per class for the checkpoints that seed labels will
                    be available for. In eval 2, this is the first 4 checkpoints.
            problem_type: str, Must be in "image_classification", "object_detection"
        """
        assert problem_type == 'image_classification' or problem_type == 'object_detection' \
            or problem_type == "video_classification", \
            "problem_type must be image_classification, object_detection or video classification"
        num_classes = lbls['class'].nunique()
        print(f"num classes is: {num_classes}")
        # Budgets are cumulative, so we calculate the number of examples to give at each stage
        tmp_budgets_per_class = [0] + budgets_per_class
        examples_per_class = [tmp_budgets_per_class[i + 1] - tmp_budgets_per_class[i] for i in
                              range(len(budgets_per_class))]
        num_examples = [b * num_classes for b in examples_per_class]
        seed_labels = {}
        id_key = 'id'
        if problem_type == "video_classification":
            lbls = self.video_remove_multilabel(lbls, num_examples[-1])
            print(f"finished removing multilabels for video")

        # Get the seed labels for each checkpoint
        print(f"getting seed labels for each checkpoint")
        for i in range(len(budgets_per_class)):
            print(f"looking for seeds at checkpoint {i}")
            # Try to take the top n from each class, and return the number of extra examples needed if a class does not
            # have enough labels to meet the budget.
            seeds, num_classes_left = self._check_seeds(lbls, examples_per_class[i], problem_type)
            extra_examples_needed = num_examples[i] - seeds[id_key].nunique()
            print(f"on checkpoint {i} got {seeds['id'].nunique()} seeds and need {extra_examples_needed}")

            # Remove the seeds from the pool of labels we can consider
            lbls = self._drop_lbls(lbls, seeds)

            # Handle the case where we did not have enough labels from one or more classes
            while (extra_examples_needed) > 0 and seeds[id_key].nunique() < num_examples[i]:
                # Try to add evenly across the classes
                seeds_per_class, remainder = divmod(extra_examples_needed, num_classes_left)
                print(f"{seeds_per_class} are needed from each of the remaining {num_classes_left} plus {remainder} seeds")
                if seeds_per_class > 0:

                    extra_seeds, num_classes_left = self._check_seeds(lbls, seeds_per_class, problem_type)
                    print(f"{seeds_per_class} are needed from each of the remaining {num_classes_left}")
                    # if problem_type == "object_detection":
                    #     extra_seeds = self._get_all_lbls(lbls, extra_seeds)

                    lbls = self._drop_lbls(lbls, extra_seeds)
                    seeds = pd.concat([extra_seeds, seeds])
                # Add the remainder
                if remainder > 0:
                    print(f"Getting remainder")
                    remainder_lbls = lbls.groupby('class').apply(lambda x: x.head(1)).head(remainder)
                    if problem_type == "object_detection":
                        remainder_lbls = self._get_all_lbls(lbls, remainder_lbls)
                    elif problem_type == 'video_classification':
                        remainder_lbls == self._get_all_vid_labels(lbls, remainder_lbls['end_frame'])
                    print(f"number of remainder labels: {remainder_lbls.shape[0]}")
                    seeds = pd.concat([seeds, remainder_lbls])
                    print(f"seeds size is now: {seeds.shape[0]}")
                    lbls = self._drop_lbls(lbls, remainder_lbls)
                extra_examples_needed = num_examples[i] - seeds[id_key].nunique()

            # Check that we have the expected number of seed labels
            assert seeds[id_key].nunique() == num_examples[i], \
                f"At checkpoint {i} there should be {num_examples[i]} unique seed ids, but there were {seeds[id_key].nunique()} generated"

            print(f"at checkpoint {i} seeds is {seeds}")

            seeds = seeds.to_dict(orient='records')
            seed_labels[str(i)] = seeds

        print(f"at end of loop seeds is: {seed_labels}")
        # Check that all the seed ids are unique at the different budgets
        for i in range(3):
            intersection = set([lbl[id_key] for lbl in seed_labels[str(i)]]).intersection(
                set([lbl[id_key] for lbl in seed_labels[str(i + 1)]]))
            assert len(intersection) == 0, \
                f"Seed labels at checkpoint {i} and {i+1} contain the same ids: {intersection}"
        return seed_labels

