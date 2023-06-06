from http.client import OK
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from data.coco_karpathy_dataset import (
    coco_karpathy_train,
    coco_karpathy_caption_eval,
    coco_karpathy_retrieval_eval,
)
from data.nocaps_dataset import nocaps_eval
from data.flickr30k_dataset import flickr30k_train, flickr30k_retrieval_eval
from data.vqa_dataset import vqa_dataset, AokVqaDataset, OkVqaDataset, ArtVQADataset, VQARephrasingsDataset, PathVqaDataset, RsvqaDataset, GenericVqaDataset
from data.nlvr_dataset import nlvr_dataset
from data.pretrain_dataset import pretrain_dataset
from data.vqg_dataset import VqgDataset, AokVqgDataset
from transform.randaugment import RandomAugment


def create_dataset(dataset, config, min_scale=0.5):

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                config["image_size"],
                scale=(min_scale, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            RandomAugment(
                2,
                5,
                isPIL=True,
                augs=[
                    "Identity",
                    "AutoContrast",
                    "Brightness",
                    "Sharpness",
                    "Equalize",
                    "ShearX",
                    "ShearY",
                    "TranslateX",
                    "TranslateY",
                    "Rotate",
                ],
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(
                (config["image_size"], config["image_size"]),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if dataset == "pretrain":
        dataset = pretrain_dataset(
            config["train_file"], config["laion_path"], transform_train
        )
        return dataset

    elif dataset == "caption_coco":
        train_dataset = coco_karpathy_train(
            transform_train,
            config["image_root"],
            config["ann_root"],
            prompt=config["prompt"],
        )
        val_dataset = coco_karpathy_caption_eval(
            transform_test, config["image_root"], config["ann_root"], "val"
        )
        test_dataset = coco_karpathy_caption_eval(
            transform_test, config["image_root"], config["ann_root"], "test"
        )
        return train_dataset, val_dataset, test_dataset

    elif dataset == "nocaps":
        val_dataset = nocaps_eval(
            transform_test, config["image_root"], config["ann_root"], "val"
        )
        test_dataset = nocaps_eval(
            transform_test, config["image_root"], config["ann_root"], "test"
        )
        return val_dataset, test_dataset

    elif dataset == "retrieval_coco":
        train_dataset = coco_karpathy_train(
            transform_train, config["image_root"], config["ann_root"]
        )
        val_dataset = coco_karpathy_retrieval_eval(
            transform_test, config["image_root"], config["ann_root"], "val"
        )
        test_dataset = coco_karpathy_retrieval_eval(
            transform_test, config["image_root"], config["ann_root"], "test"
        )
        return train_dataset, val_dataset, test_dataset

    elif dataset == "retrieval_flickr":
        train_dataset = flickr30k_train(
            transform_train, config["image_root"], config["ann_root"]
        )
        val_dataset = flickr30k_retrieval_eval(
            transform_test, config["image_root"], config["ann_root"], "val"
        )
        test_dataset = flickr30k_retrieval_eval(
            transform_test, config["image_root"], config["ann_root"], "test"
        )
        return train_dataset, val_dataset, test_dataset

    elif dataset == "vqa":
        train_dataset = vqa_dataset(
            transform_train,
            config["ann_root"],
            config["vqa_root"],
            config["vg_root"],
            train_files=config["train_files"],
            split="train",
            truncate_to=config.truncate_train_dataset_to,
        )
        test_dataset = vqa_dataset(
            transform_test,
            config["ann_root"],
            config["vqa_root"],
            config["vg_root"],
            split="test",
        )
        return train_dataset, test_dataset

    elif dataset == "vqg":
        # We can use these datasets in mostly the same way
        # except that okvqa doesn't have rationales.
        if config.dataset_name in ("aokvqa", "okvqa", "artvqa", "pathvqa"):
            split = "val" if config.use_validation_set_as_test_set else "test"
            train_dataset = AokVqgDataset(
                transform_train,
                config["ann_root"],
                config["vqa_root"],
                config["vg_root"],
                train_files=config["train_files"],
                split="train",
                truncate_to=config.truncate_train_dataset_to,
                use_rationale=config["use_rationale"],
                generate_rationale_first=config.generate_rationale_first,
            )
            test_dataset = AokVqgDataset(
                transform_test,
                config["ann_root"],
                config["vqa_root"],
                config["vg_root"],
                split=split,
            )
        elif config.dataset_name == "vqa":
            train_dataset = VqgDataset(
                transform_train,
                config["ann_root"],
                config["vqa_root"],
                config["vg_root"],
                train_files=config["train_files"],
                split="train",
                truncate_to=config.truncate_train_dataset_to,
            )
            test_dataset = vqa_dataset(
                transform_test,
                config["ann_root"],
                config["vqa_root"],
                config["vg_root"],
                split="test",
            )
        else:
            raise ValueError("Unknown dataset name: {}".format(config.dataset_name))
        return train_dataset, test_dataset

    elif dataset == "nlvr":
        train_dataset = nlvr_dataset(
            transform_train, config["image_root"], config["ann_root"], "train"
        )
        val_dataset = nlvr_dataset(
            transform_test, config["image_root"], config["ann_root"], "val"
        )
        test_dataset = nlvr_dataset(
            transform_test, config["image_root"], config["ann_root"], "test"
        )
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'aokvqa':
        train_dataset = AokVqaDataset(
            transform_train,
            config["ann_root"],
            config["vqa_root"],
            config["vg_root"],
            train_files=config["train_files"],
            split="train",
            truncate_to=config.truncate_train_dataset_to,
            append_rationale_to_answer=config.append_rationale_to_answer,
            append_rationale_to_question=config.append_rationale_to_question,
            use_rationale_as_answer=config.use_rationale_as_answer,
        )
        if config.use_validation_set_as_test_set:
            test_dataset = AokVqaDataset(
                transform_test,
                config["ann_root"],
                config["vqa_root"],
                config["vg_root"],
                split="val",
                # We might want to use generated rationales at test time.
                append_rationale_to_question=config.append_rationale_to_question,
            )
        else:
            test_dataset = AokVqaDataset(
                transform_test,
                config["ann_root"],
                config["vqa_root"],
                config["vg_root"],
                split="test",
            )
        return train_dataset, test_dataset
    elif dataset == 'okvqa':
        train_dataset = OkVqaDataset(
            transform=transform_train,
            ann_root=config["ann_root"],
            vqa_root=config["vqa_root"],
            train_files=config["train_files"],
            split="train",
            truncate_to=config.truncate_train_dataset_to,
        )
        # There's no test set for OKVQA, so we
        # use the validation set as the test set.
        test_dataset = OkVqaDataset(
            transform_test,
            ann_root=config["ann_root"],
            vqa_root=config["vqa_root"],
            split="val",
        )
        return train_dataset, test_dataset
    elif dataset == 'artvqa':
        train_dataset = ArtVQADataset(
            transform_train,
            config["ann_root"],
            config["vqa_root"],
            train_files=config["train_files"],
            split="train",
            truncate_to=config.truncate_train_dataset_to,
        )
        test_dataset = ArtVQADataset(
            transform_test,
            config["ann_root"],
            config["vqa_root"],
            split="test",
        )
        return train_dataset, test_dataset
    elif dataset == 'pathvqa':
        train_dataset = PathVqaDataset(
            transform_train,
            config["ann_root"],
            config["vqa_root"],
            train_files=config["train_files"],
            split="train",
            truncate_to=config.truncate_train_dataset_to,
        )
        test_dataset = PathVqaDataset(
            transform_test,
            config["ann_root"],
            config["vqa_root"],
            split="test",
        )
        return train_dataset, test_dataset
    elif dataset == 'vqa_rephrasings':
        train_dataset = VQARephrasingsDataset(
            transform=transform_train,
            ann_root=config["ann_root"],
            vqa_root=config["vqa_root"],
            train_files=config["train_files"],
            split="train",
            truncate_to=config.truncate_train_dataset_to,
        )
        # There's no test set for VQA Rephrasings, so we
        # use the validation set as the test set.
        test_dataset = VQARephrasingsDataset(
            transform_test,
            ann_root=config["ann_root"],
            vqa_root=config["vqa_root"],
            split="val",
        )
        return train_dataset, test_dataset
    elif dataset == 'rsvqa':
        train_dataset = RsvqaDataset(
            transform=transform_train,
            ann_root=config["ann_root"],
            vqa_root=config["vqa_root"],
            train_files=config["train_files"],
            split="train",
            truncate_to=config.truncate_train_dataset_to,
        )

        test_dataset = RsvqaDataset(
            transform_test,
            ann_root=config["ann_root"],
            vqa_root=config["vqa_root"],
            split="test",
        )
        return train_dataset, test_dataset
    elif dataset == 'generic_vqa':
        train_dataset = GenericVqaDataset(
            transform=transform_train,
            ann_root=config["ann_root"],
            vqa_root=config["vqa_root"],
            ann_files=config["train_files"],
            split="train",
            truncate_to=config.truncate_train_dataset_to,
            answer_list=config["answer_list"],
        )
        test_dataset = GenericVqaDataset(
            transform_test,
            ann_root=config["ann_root"],
            vqa_root=config["vqa_root"],
            ann_files=[config["val_file"]],
            split="val",
            answer_list=config["answer_list"],
        )
        return train_dataset, test_dataset
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
        )
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(
        datasets, samplers, batch_size, num_workers, is_trains, collate_fns
    ):
        if is_train:
            shuffle = sampler is None
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders
