# The following script is used to set the data for ASR model which includes:
# 1) preparing the character vocabulary
# 2) tokenize the character vocabulary
# 2) assigning the special words to vocabulary
# 3) preparing the feature extractor and Wav2Vec2 Processor

# torchaudio,jiwer
import os,argparse,json,torchaudio
from datasets import Dataset
import pandas as pd
from transformers import Wav2Vec2CTCTokenizer,Wav2Vec2FeatureExtractor,Wav2Vec2Processor

parser=argparse.ArgumentParser(description="The following arguments are required to set the data for model building",
                               epilog="",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--train-data-path", type=str, default="", required=True,
                    help="")

parser.add_argument("--train-filename", type=str, default="", required=True,
                    help="")

parser.add_argument("--special-token-path", type=str, default="", required=True,
                    help="")

parser.add_argument("--export-path", type=str, default="", required=True,
                    help="Specify the path where all the data will be exported")

args=parser.parse_args()

special_tokens=["|","[UNK]","[PAD]"]

def resample_audio(audio_data):
    """
    The following function is used to resample the audio file 
    to 16000 for making it available for model training
    """
    for audio in os.listdir(f"{audio_data}/wav.scp"):
        audio_path=audio.split(" ")[1]
        speech_array, sampling_rate = torchaudio.load(f"{audio_path}")
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        resampled_array = resampler(speech_array).squeeze().numpy()

def extract_characters(batch):
    """
    The following function is used to:
    1) extract all the characters from a transcript
    2) map the vocab data to the same transcript in csv dataset
    """
    char_data=" ".join(batch["transcript"])
    char_vocab=list(set(char_data))
    return {"Vocabulary":[char_vocab], "Characters":[char_data]}

def data_processing_tokenizing(vocab_file,model_path):
    """
    The follwoing function is used to tokenize and process the data using
    Wav2vecTokenizer and Wav2VecProcessor
    """
    tokenizer=Wav2Vec2CTCTokenizer(vocab_file=vocab_file,unk_token="[UNK]",pad_token="[PAD]",word_delimiter_token="|")
    featurizer=Wav2Vec2FeatureExtractor(feature_size=1,sampling_rate=16000,padding_value=0.0,do_normalize=True,return_attention_mask=True)
    processor=Wav2Vec2Processor(feature_extractor=featurizer,tokenizer=tokenizer)
    processor.save_pretrained(model_path)
    return ("The pretrained model parameters are saved in [ {}] ".format(model_path))

def process_data(train_data,special_values,save_path):
    """
    The folowing function is used to process the audio dataset by following steps:
    1) Set all the data in one csv file
    2) Extract the characters out of it and prepare a vocabulary of characters and assign some special variables
    3) Tokenize each of the characters in vocabulary
    4) Prepare feature extractor of Wav2Vec to extract all features out of the audios
    5) Prepare the Wav2Vec processor and parse the path where to save the model
    """
    # preparing the dataset
    dataset=pd.DataFrame()
    filename="train_model_data.csv"
    
    open_train_data=pd.read_csv(f"{train_data}/wav.scp",header=None)
    open_train_text=pd.read_csv(f"{train_data}/text",header=None)
    open_train_data.columns=['label']
    open_train_text.columns=['transcript']
    transcription_list=[" ".join(transcript[2:]) for transcript in open_train_text["transcript"].str.split(" ")]
    dataset[['audio_id']]=pd.DataFrame(open_train_data["label"].str.split(" ",expand=True)[0])
    dataset[['audio_path']]=pd.DataFrame(open_train_data["label"].str.split(" ",expand=True)[1])
    dataset[['transcript']]=pd.DataFrame(transcription_list)
    dataset[['sample_rate']]=pd.DataFrame(open_train_data["label"].str.split(" ",expand=True)[2])
    dataset[['duration']]=pd.DataFrame(open_train_data["label"].str.split(" ",expand=True)[3])
    dataset.to_csv(f"{save_path}/{filename}") if filename not in os.listdir(f"{save_path}") else print("The training csv file already exist")
    # prepare the vocabulary of characters out of transcription file
    dataset=Dataset.from_pandas(dataset)
    character_vocabulary=dataset.map(extract_characters, batched=True,batch_size=-1,keep_in_memory=True,remove_columns=dataset.column_names)
    vocab_list=list(set(character_vocabulary["Vocabulary"][0]))
    vocab_dict={v:k for k,v in enumerate(vocab_list)}
    print(vocab_dict)
    # add special characters in vocabulary
    vocab_dict["|"]=vocab_dict[' ']
    del vocab_dict[' ']
    vocab_dict['[UNK]']=len(vocab_dict)
    vocab_dict['[PAD]']=len(vocab_dict)
    # write_vocab_file=open(f"{save_path}/char_vocab.json", 'w',encoding="UTF-8") if "char_vocab.json" not in os.listdir(f"{save_path}") else print("The vocab file already exist")
    with open(f"{save_path}/char_vocab.json", 'w',encoding="utf-8") as vocab_file:
        json.dump(vocab_dict,vocab_file)
    tokenize_data=data_processing_tokenizing(f"{save_path}/char_vocab.json",f"{save_path}/model_V1")
    
    
if "__main__" == __name__:
    # train_data_path="/home/samarth/testing/extra_testing/local_salesphony_testing/salesphony_testing/Model/db/mapping/general/asr"
    # exp_path="/home/samarth/testing/extra_testing/local_salesphony_testing/salesphony_testing/Model/ASR/exp/model_data"
    process_data(args.train_data_path,args.special_token_path,args.export_path)