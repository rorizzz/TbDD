import pandas as pd
from datasets import Dataset, DatasetDict
import librosa
from tqdm import tqdm
import numpy as np
from datasets import Audio
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor

import warnings
warnings.filterwarnings('ignore')
import setproctitle
setproctitle.setproctitle("Whisper")

# Datasets
def load_audio_data(file_path):
    audio_array, sampling_rate = librosa.load(file_path, sr=None)
    return audio_array, sampling_rate


def create_dataset(csv_file):
    df = pd.read_csv(csv_file, nrows=80000) # , nrows=40000

    processed_data = []
    # for _, row in df.iterrows():
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing audio files"):
        audio_path = row['audio_path']
        label = row['label']
        audio_array, sampling_rate = load_audio_data(audio_path)
        
        item = {
            'audio': {
                'path': audio_path,
                'array': audio_array,
                'sampling_rate': sampling_rate
            },
            'sentence': label
        }
        processed_data.append(item)
    
    dataset = Dataset.from_pandas(pd.DataFrame(processed_data))
    train_dataset, test_dataset = dataset.train_test_split(test_size=0.05).values()
        
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    return dataset_dict


csv_file = 'data_preprocess/new/co/word/VCTK_token_word.csv'
dataset = create_dataset(csv_file)

print('loading dataset....')
print(dataset)


import torchaudio
def downsample_audio_torchaudio(example):
    waveform, sr = torchaudio.load(example['audio']['path'])
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    resampled_waveform = resampler(waveform)
    example['audio']['array'] = resampled_waveform.numpy()
    example['audio']['sampling_rate'] = 16000
    return example


def audio_torchaudio(example):
    waveform, sr = torchaudio.load(example['audio']['path'])
    # resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    # resampled_waveform = resampler(waveform)
    example['audio']['array'] = waveform.numpy()
    example['audio']['sampling_rate'] = 16000
    return example



# print('downsampling...')
# dataset = dataset.map(
#     audio_torchaudio, 
#     num_proc=2,
#     load_from_cache_file=False
# )

# print('finish downsampling')



# WhisperProcessr
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="en", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="en", task="transcribe")


## add dys tokens
new_tokens = ["[REP]", "[DEL]", "[PAU]", "[INS]"]
new_tokens_phn = ["[REP]", "[DEL]", "[PRO]", "[SUB]", "jh", "dh"]
 
tokenizer.add_tokens(list(new_tokens))


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch



print('preparing dataset...')
dataset = dataset.map(
    prepare_dataset, 
    remove_columns=dataset.column_names["train"], 
    num_proc=2, #4
    load_from_cache_file=False
)


from transformers import WhisperForConditionalGeneration
from transformers import WhisperConfig
from transformers import GenerationConfig
import torch
from safetensors.torch import load_file


MODEL_PARAMS = 'pre' # [pre, init, exist]

print('loading pretrained model...')
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
# resize model
model.resize_token_embeddings(len(tokenizer))

if MODEL_PARAMS == 'init':
    print("init params...")
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Conv1d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    model.apply(init_weights) # init params

elif MODEL_PARAMS == 'exist':
    print('Loading custom model parameters...')
    state_dict = load_file('/data/xuanru/whisper/whisper-small-vctktoken-single-word-pause/checkpoint-8000/model.safetensors')
    model.load_state_dict(state_dict, strict=False)


# print('No pretrained model, create one....')
# whisper_config = WhisperConfig.from_pretrained('openai/whisper-small')
# model = WhisperForConditionalGeneration(whisper_config)


# model.generation_config.lang_to_id = config_data['lang_to_id']
# model.generation_config.alignment_heads = config_data['alignment_heads']
# model.generation_config.is_multilingual = config_data['is_multilingual']
# model.generation_config.task_to_id = config_data['task_to_id']

model.generation_config.language = "<|en|>"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None
model.generation_config.use_cache = False


## modify supperes ids for < > [ ]
import json
config_path = 'config.json'
generation_path = 'generation_config.json'

with open(config_path, 'r', encoding='utf-8') as file:
    config_data = json.load(file)

with open(generation_path, 'r', encoding='utf-8') as file:
    generation_config_data = json.load(file)

model.config.suppress_tokens = config_data['suppress_tokens']
model.generation_config.suppress_tokens = generation_config_data['suppress_tokens']
model.config.begin_suppress_tokens = config_data['begin_suppress_tokens']
model.generation_config.begin_suppress_tokens = generation_config_data['begin_suppress_tokens']


from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)


import evaluate
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    # print(f"\nWER: {wer}%")
    return {"wer": wer}



from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="/data/xuanru/whisper/co-new/pre-vctktoken-co-subset-word-token",  # change to a repo name of your choice
    per_device_train_batch_size=8, # 16 1 4 4
    gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500, # 500
    max_steps=10000, # 4000
    gradient_checkpointing=False, # True
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8, # 8 4
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=200, # 500
    eval_steps=200, # 500 
    logging_steps=100, # 25
    # report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    dataloader_num_workers=2
)


from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)

print('start training...')
trainer.train()