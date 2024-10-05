import librosa
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, DatasetDict
import torchaudio
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import WhisperTokenizer
from safetensors.torch import load_file
from tqdm import tqdm
import torch


cuda_device = f"cuda:7"
device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')

# Datasets
def load_audio_data(file_path):
    audio_array, sampling_rate = librosa.load(file_path, sr=None)
    return audio_array, sampling_rate


def create_dataset(csv_file):
    df = pd.read_csv(csv_file, nrows=1000) # , nrows=40000

    processed_data = []
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
    train_dataset, test_dataset = dataset.train_test_split(test_size=0.01).values()
        
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

    return dataset_dict



csv_file = 'data_preprocess/new/phn/VCTK_token_phn_subset.csv'
dataset = create_dataset(csv_file)

print('loading dataset....')
print(dataset)


processor = WhisperProcessor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="en", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

new_tokens = ["[REP]", "[DEL]", "[PAU]", "[INS]"]
new_tokens_phn = ["[REP]", "[DEL]", "[PRO]", "[SUB]", "jh", "dh"]
tokenizer.add_tokens(list(new_tokens_phn))
model.resize_token_embeddings(len(tokenizer))

print("loading model weights....")

data_folder = '/data/xuanru/whisper/new'
model_name = 'pre-vctktoken-single-subset-phn-token'
checkpoint_index = '10000'

state_dict = load_file(f'{data_folder}/{model_name}/checkpoint-{checkpoint_index}/model.safetensors')
model.load_state_dict(state_dict, strict=False)
model.to(device)

# start testing

output_file = open('test_result/test_phn.txt', 'w')  
length = len(dataset['train'])

for i in tqdm(range(length)):
    audio_sample = dataset['train'][i]['audio']
    sentence = dataset['train'][i]['sentence']
    path = audio_sample['path']

    input_features = processor(
        audio_sample["array"], sampling_rate=audio_sample["sampling_rate"], return_tensors="pt").input_features
    
    input_features = input_features.to(device, torch.float32)

    # input_ids = tokenizer(sentence).input_ids 

    predicted_ids = model.generate(input_features)

    transcription = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    # print(f"{path}: predict: {transcription} |sentence: {sentence}")

    output = f"{path}: {transcription}|{sentence}\n"

    output_file.write(output)
    output_file.flush()

        

