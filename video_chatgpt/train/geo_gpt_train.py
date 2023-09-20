import copy
from dataclasses import dataclass, feild
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence
import torch
import transformers
from torch.utils.data import Dataset
from video_chatgpt.train.llava_trainer import VideoChatGPTTrainer
from video_chatgpt import video_conversation as conversation_lib
from video_chatgpt.model import *
import torch.distributed as dist
from video_chatgpt.constants import *
import pickle
import fire



IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class  ModelArguments:
    model_name_or_path: Optional[str] = feild(default="facebook/opt-125")

def train():
    parser = transformers.HfArgumentParser(
        
    )
