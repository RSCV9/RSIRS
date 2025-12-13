import os
import sys
import torch.utils.data as data
import torch
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import random

from bert.tokenization_bert import BertTokenizer

import h5py
from refer.refer import REFER
import json
from args import get_parser

# Dataset configuration initialization
parser = get_parser()
args = parser.parse_args()


def add_random_boxes(img, min_num=20, max_num=60, size=32):
    h,w = size, size
    img = np.asarray(img).copy()
    img_size = img.shape[1]
    boxes = []
    num = random.randint(min_num, max_num)
    for k in range(num):
        y, x = random.randint(0, img_size-w), random.randint(0, img_size-h)
        img[y:y+h, x: x+w] = 0
        boxes. append((x,y,h,w) )
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    return img


class ReferDataset(data.Dataset):

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        self.classes = []
        self.image_transforms = image_transforms
        self.target_transform = target_transforms
        self.split = split
        self.refer = REFER(args.refer_data_root, args.dataset, args.splitBy)
        ########################
        image_index = set()
        ########################
        self.max_tokens = 20
        self.args = args

        self.ref_index = []
        # with open('/home/lm/Desktop/HY_OV_spanRS_data.json', 'r', encoding='utf-8') as f:
        with open(args.dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        for index, i in enumerate(self.data):
            if split == i['split']:
                self.ref_index.append(index)
                image_index.add(i['file_name'])
                # self.mask_file.append(i['mask'])
        image_index = list(image_index)
        refs_index = self.ref_index


        num_images_to_mask = int(len(image_index) * 0.1)
        self.images_to_mask = random.sample(image_index, num_images_to_mask)



        self.input_ids = []
        self.attention_masks = []
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

        self.eval_mode = eval_mode


    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.ref_index)

    def __getitem__(self, index):
        data_index = self.ref_index[index]
        image_path = self.data[data_index]['file_name']
        mask_path = self.data[data_index]['mask']
        language = self.data[data_index]['sentence']
        # this_ref_id = self.ref_ids[index]
        # this_img_id = self.refer.getImgIds(this_ref_id)
        # this_img = self.refer.Imgs[this_img_id[0]]
        root_dir = '/media/lm/lmssd1/1_HY_dataset/image'
        root_mask_dir = '/media/lm/lmssd1/1_HY_dataset/mask'
        img = Image.open(os.path.join(root_dir, image_path))
        if self.split == 'train' and image_path in self.images_to_mask:
            img = add_random_boxes(img)

        # ref = self.refer.loadRefs(this_ref_id)
        ref_mask = Image.open(os.path.join(root_mask_dir, mask_path))
        ref_mask = np.array(ref_mask)
        # max_value = np.max(ref_mask)
        # unnique = np.unique(ref_mask)
        annot = np.zeros(ref_mask.shape)
        annot[ref_mask == 255] = 1
        # max_value1 = np.max(annot)
        # unnique1 = np.unique(annot)
        annot = Image.fromarray(annot.astype(np.uint8), mode="P")

        if self.image_transforms is not None:
            # resize, from PIL to tensor, and mean and std normalization
            img, target = self.image_transforms(img, annot)

        # langeuage_index = self.map['{}'.format(index)]
        ##################
        sentences_for_ref = []
        attentions_for_ref = []

        # for i, el in enumerate(ref):
        sentence_raw = self.data[data_index]['sentence']
        attention_mask = [0] * self.max_tokens
        padded_input_ids = [0] * self.max_tokens

        input_ids = self.tokenizer.encode(text=sentence_raw, add_special_tokens=True)

        # truncation of tokens
        input_ids = input_ids[:self.max_tokens]

        padded_input_ids[:len(input_ids)] = input_ids
        attention_mask[:len(input_ids)] = [1] * len(input_ids)

        sentences_for_ref.append(torch.tensor(padded_input_ids).unsqueeze(0))
        attentions_for_ref.append(torch.tensor(attention_mask).unsqueeze(0))
        ##################
        if self.eval_mode:
            embedding = []
            att = []
            for s in range(len(sentences_for_ref)):
                e = sentences_for_ref[s]
                a = attentions_for_ref[s]
                embedding.append(e.unsqueeze(-1))
                att.append(a.unsqueeze(-1))

            tensor_embeddings = torch.cat(embedding, dim=-1)
            attention_mask = torch.cat(att, dim=-1)
        else:
            choice_sent = np.random.choice(len(sentences_for_ref))
            tensor_embeddings = sentences_for_ref[choice_sent]
            attention_mask = attentions_for_ref[choice_sent]
        if self.args.ov == 1:
            # print(self.data[data_index])
            ov = self.data[data_index]['OV']
            # cls = self.data[data_index]['category']
            domain = self.data[data_index]['domain']
            return img, target, tensor_embeddings, attention_mask, image_path, language, mask_path, ov, domain, data_index
        else:
            ov = 0
            domain = self.data[data_index]['domain']
            # cls = self.data[data_index]['category']
            return img, target, tensor_embeddings, attention_mask, image_path, language, mask_path, ov, domain, data_index