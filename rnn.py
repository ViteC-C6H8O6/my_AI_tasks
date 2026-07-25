import torch as tch
from torch.utils.data import *
import torch.nn as nn 
import sentencepiece as spm
spm.SentencePieceTrainer.Train('--input="data\\en2cn\\train_en.txt" --model_prefix=en_bpe --vocab_size=16000 --model_type=bpe --character_coverage=1.0 --unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3')
spm.SentencePieceTrainer.Train('--input="C:\\Users\\wang\\Desktop\\学习文件\\en2cn\\train_zh.txt" --model_prefix=zh_bpe --vocab_size=16000 --model_type=bpe --character_coverage=0.9995 --unk_id=0 --pad_id=1 --bos_id=2 --eos_id=3')


