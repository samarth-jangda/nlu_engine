# The following script is used to build the complete ASR model and save in respective directory. It prepares following:
# 1) convert speech to array
# 2) prepare a data collator and data paddor
# 3) then set the model parameters,pretrained model data params and start the model training

# python3 ASR/src/model.py --pretrained-processor-path=/home/samarth/testing/salesphony_testing/Model/ASR/exp/model_V1/model_V1/ --model-config-path=/home/samarth/testing/salesphony_testing/Model/ASR/exp/tran_model.conf --train-csv-path=/home/samarth/testing/salesphony_testing/Model/ASR/InputData/train/train_model_data.csv --resampling-frequency=16000 --test-csv-path=/home/samarth/testing/salesphony_testing/Model/ASR/InputData/test/test_model.csv --save-model-path=/home/samarth/testing/salesphony_testing/Model/ASR/exp/model_V1/HindiASRModel 

import argparse,librosa,torch,torchaudio,random
from datasets import load_metric
from typing import Dict, List, Optional, Union
import IPython.display as ipd
import numpy as np
from dataclasses import dataclass
from datasets import Dataset
import pandas as pd
from transformers import  Wav2Vec2ForCTC,TrainingArguments,Wav2Vec2Processor,Trainer

parser=argparse.ArgumentParser(description="The following arguments are used to retrain the ASR Wav2Vec model",
                               epilog="",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter) 

parser.add_argument("--pretrained-processor-path", type=str, default="", required=True,
                    help="Specify the path of pretrained model processor")

parser.add_argument("--model-config-path", type=str, default="", required=True,
                    help="Specify the model configuration file path")

parser.add_argument("--train-csv-path", type=str, default="", required=True,
                    help="Specify the path of training csv file path")

parser.add_argument("--resampling-frequency", type=str, default="", required=True,
                    help="Specify the frequecy to which the audio will be resampled")

parser.add_argument("--test-csv-path", type=str, default="", required=True,
                    help="Specify the path of testing csv file for model evaluation")

parser.add_argument("--save-model-path", type=str, default="", required=True,
                    help="Specify the path where to save the model")

args=parser.parse_args()

processor_model_path=args.pretrained_processor_path
model_config=args.model_config_path
training_audio_path=args.train_csv_path
resampling_frequency=args.resampling_frequency
testing_data=args.test_csv_path
save_model_path=args.save_model_path

# def process_model_data(processor_model_path,model_config,training_audio_path):
#     """
#     The following function is used to process the bulding of ASR model
#     and saving in the corresponding path for evaluation and testing
#     """
#     global compute_model_metrics
#     # global data_collator
#     global preparing_dataset
#     global processor

    
def resample_and_speech_to_array(batch,resampling_frequency):
    """
    The following function is used to resample the audio
    to a specified rate
    """
    speech_array, sample_rate=torchaudio.load(batch["audio_path"])
    print(sample_rate)
    batch["speech"]=speech_array[0].numpy()
    batch["speech"]=librosa.resample(np.asarray(batch["speech"]),orig_sr=8000,target_sr=16_000)
    batch["sampling_rate"]=16000
    batch["target_text"]=batch["transcript"]
    return (batch)
    
def display_data(batch):
    """
    The following fucntion is used to visualize the dataset, to
    validate of it is correctly prepared
    """
    rand_int=random.randint(0,len(batch))
    ipd.Audio(data=np.asarray(batch[rand_int]["speech"]),autoplay=True,rate=resampling_frequency)
    print("Transcript:",batch[rand_int]["target_text"])
    print("InputSpeechShape:",np.asarray(batch[rand_int]["speech"]).shape)
    print("SamplingRate:",batch[rand_int]["sampling_rate"])
    
def prepare_data_using_processor(batch):
    """
    The following function is used to process the data via
    Wav2Vec Processor
    """
    assert (
        len(set(batch["sampling_rate"])) == 1
    ),f"Warning:Not all the audios have same sampling rate : {model_processor.feature_extractor.sampling_rate}"
    batch["input_values"]=model_processor(batch["speech"],sampling_rate=batch["sampling_rate"][0]).input_values
    with model_processor.as_target_processor():
        batch["labels"]=model_processor(batch["target_text"]).input_ids
    return (batch)    
    
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """
    processor:Wav2Vec2Processor
    padding: Union[bool, str]= True
    max_length: Optional[int]=None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    
    def __call__(self, features: List[Dict[str,Union[List[int],torch.Tensor]]]) -> Dict[str,torch.Tensor]:
        input_features=[{"input_values": feature["input_values"]} for feature in features]
        label_features=[{"input_ids": feature["labels"]} for feature in features]
        
        batch=self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",   
        )
        with self.processor.as_target_processor():
            labels_batch=self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )
        print(labels_batch)    
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch['labels']=labels
        return (batch)    

# def train_model(train_dataset,data_collator,test_dataset,processor,model_config,save_model_path):
"""
The following function is used to call:
1) Call the pretrained processor model
2) Call the model training arguments
3) Call the trainer to start the training
"""
word_error_rate_metric=load_metric("wer")
def compute_wer(preds):
    """
    The following function is used to compute the
    final Word Error Rate of the trained model.
    """
    pred_logits = preds.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    preds.label_ids[preds.label_ids == -100] = model_processor.tokenizer.pad_token_id
    pred_str = model_processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = model_processor.batch_decode(preds.label_ids, group_tokens=False)
    wer = word_error_rate_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

model_processor=Wav2Vec2Processor.from_pretrained(processor_model_path)
# print(processor)
train_csv_data=pd.read_csv(training_audio_path)
train_data=Dataset.from_pandas(train_csv_data)
train_new_csv_data=train_data.map(resample_and_speech_to_array,16000,remove_columns=train_data.column_names)
print(train_new_csv_data.column_names)
# visualize_train_data=display_data(train_csv_data)
preparing_dataset=train_new_csv_data.map(prepare_data_using_processor,remove_columns=train_new_csv_data.column_names,batched=True)
data_collator_class=DataCollatorCTCWithPadding(processor=model_processor,padding=True)

model=Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53",
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    gradient_checkpointing=True,
    ctc_loss_reduction="mean",
    pad_token_id=model_processor.tokenizer.pad_token_id,
    vocab_size=len(model_processor.tokenizer)
)
model.freeze_feature_extractor()
# the following arguments must be from model_config file but for now we are just setting for testing
training_args=TrainingArguments(
    output_dir=save_model_path,
    group_by_length=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    num_train_epochs=30,
    fp16=False,
    save_steps=400,
    eval_steps=400,
    logging_steps=400,
    learning_rate=1e-4,
    warmup_steps=500,
    save_total_limit=1,
)
trainer = Trainer(
model=model,
data_collator=data_collator_class,
args=training_args,
compute_metrics=compute_wer,
train_dataset=preparing_dataset,
eval_dataset=testing_data,
tokenizer=model_processor.feature_extractor,
)
trainer.train()
    
# if "__main__" == __name__:
#     processor_model_path=args.pretrained_processor_path
#     model_config=args.model_config_path
#     training_audio_path=args.train_csv_path
#     resampling_frequency=args.resampling_frequency
#     testing_data=args.test_csv_path
#     save_model_path=args.save_model_path
#     process_model_data(processor_model_path,model_config,training_audio_path)    
