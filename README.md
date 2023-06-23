# LwLL Scripts

This repository was for maintainers of the lwll_api codebase and not for TA1 performers.

It contains scripts that can alter user accounts as well as generate the Task metadata that was used for evaluation.

You will notice that we attach metadata to the accounts tht are created in `UserAccounts.ipynb` that have permission tiers. These permission tiers can be used for exposing only certain functionality from the web interface.

# Running Locally
This repo was modified to be able to run locally without a need for S3 or firebase.

## Running Without Firebase
In order to use these util scripts locally without firebase, all calls to the `add_budget_labels_to_task` function should have a `local_metadata_path` parameter specified with the absolute path to a folder containing subfolders for each dataset (named exactly to match each dataset) and a `metadata.json` file inside of each subfolder. An example of this structure is as follows:
```
| /Users/test/root_folder
|  | /Users/test/root_folder/cifar100
|  |  | /Users/test/root_folder/cifar100/metadata.json
|  | /Users/test/root_folder/domain_net-clipart
|  |  | /Users/test/root_folder/domain_net-clipart/metadata.json
|  | ...
```

## Running Without S3
In order to use these util scripts locally without S3, a `root_folder` parameter must be passed to the `SeedLabelGenerator` class upon creation (ex. `seed_generator = SeedLabelGenerator(root_folder="/Users/test/labels_folder")`). Passing this parameter will cause all subsequent requests for labels to look inside this root folder. The root folder should be structured to contain a folder called `live` with a subfolder inside that called `labels`. Inside this `labels` folder, there should be a folder per dataset (with an exact match name) and inside each dataset folder, there should be a `sample` or `full` folder (depending on what's requested). These folders should then contain a `labels_train.feather` file. An example of the file structure is as follows:
```
| /Users/test/labels_folder
|  | /Users/test/labels_folder/live
|  |  | /Users/test/labels_folder/live/labels
|  |  |  | /Users/test/labels_folder/live/labels/cifar100
|  |  |  |  | /Users/test/labels_folder/live/labels/cifar100/full
|  |  |  |  |  | /Users/test/labels_folder/live/labels/cifar100/full/labels_train.feather
|  |  |  |  | /Users/test/labels_folder/live/labels/cifar100/sample
|  |  |  |  |  | /Users/test/labels_folder/live/labels/cifar100/sample/labels_train.feather
|  |  |  | /Users/test/labels_folder/live/labels/domain_net-clipart
|  |  |  |  | /Users/test/labels_folder/live/labels/domain_net-clipart/full
|  |  |  |  |  | /Users/test/labels_folder/live/labels/domain_net-clipart/full/labels_train.feather
|  |  |  |  | /Users/test/labels_folder/live/labels/domain_net-clipart/sample
|  |  |  |  |  | /Users/test/labels_folder/live/labels/domain_net-clipart/sample/labels_train.feather
|  |  |  | ...
```

Alternatively, if you would still like to use S3, you can specific an AWS profile to the seed generator by using the `profile_name` parameter like so:
```
seed_generator = SeedLabelGenerator(profile_name="your_profile_name")
```
If neither a root folder or profile is specified, the generator will default to looking for an AWS profile named `saml-pub`.