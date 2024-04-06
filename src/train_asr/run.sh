# The following script is the main script which automates the setting of data for model and 
# build the model . Along with building the model the following script also validates the same

# calling the path file
. ./path.sh

# 1) Splitting the data under train and test
# calling the respective parameters
audio_path="$data_path/mapping/general/asr"
train_split_percentage=70
save_split_data_path="$asr_path/InputData"
preprocessor_name="model_V1"

# model configurations and parameters
train_csv_filename="train_model.csv"
test_csv_filename="test_model.csv"
processor_path="$asr_path/exp/model_data/model_v1"
resamping_frequency=16000
model_save_path="$asr_path/exp/model_data/model_v1/HindiASRModel"

[ ! -d "$asr_path/InputData/train" ] && echo "Since there is no train directory exist hence preparing it" && mkdir "$asr_path/InputData/train"
[ ! -d "$asr_path/InputData/test" ] && echo "Since there is no test directory exist hence preparing it" && mkdir "$asr_path/InputData/test"

# splitting the data in train and test
python3 ASR/train_test.py --audio-data-path=$audio_path --training-percentage=$train_split_percentage --saving-path=$save_split_data_path
# validating the prepared files
[ ! -f "$save_split_data_path/train/wav.scp" ] || [ ! -f "$save_split_data_path/test/wav.scp" ] && echo "There is no wav.scp found in train or test directory. Please check again" && break 1
[ ! -f "$save_split_data_path/train/transcription.txt" ] || [ ! -f "$save_split_data_path/test/transcription.txt" ] && echo "There is no transcription.txt found in train or test directory. Please check again" && break 1
num_wav_lines=$(wc -l < "$save_split_data_path/train/wav.scp" )
num_text_lines=$(wc -l < "$save_split_data_path/train/transcription.txt")
[ ! $num_wav_lines == $num_text_lines ] && echo "The audio and transcription data file size differ" && break 1
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

[ ! -d "$asr_path/exp" ] && echo "No export directory found hence preparing it" && mkdir "$asr_path/exp"
[ ! -d "$asr_path/exp/model_data" ] && echo "Making the data directory to export set data" && mkdir "$asr_path/exp/model_data" && mkdir "$asr_path/exp/model_data/$preprocessor_name"

# 2) tokenizing and preparing the data as csv file
[ ! -f "$asr_path/exp/special_token.conf" ] && echo "There is no configuration file of special tokens exist " && touch "$asr_path/exp/special_token.conf"
# saving data of special tokens
cat > $asr_path/exp/special_token.conf <<EOF
delimeter_token  |
unknown_token   [UNK]
padding_token   [PAD]
EOF    
python3 ASR/src/asr_processor.py --train-data-path="$asr_path/InputData/train" --train-filename=$train_csv_filename --special-token-path="$asr_path/exp/special_token.conf" --export-path="$processor_path"

# 3) The second step will prepare a model processor and save the same to $processor_path this path which we will use in our final step
# setting training parameters
cat > $asr_path/exp/train_model.conf <<EOF
group_by_length              True 
per_device_train_batch_size  16
gradient_accumulation_steps   2
evaluation_strategy          "steps"
num_train_epochs             50
fp16                         True
save_steps                   400
eval_steps                   400
logging_steps                400
learning_rate                0.0001
warmup_steps                 500
save_total_limit             1
EOF

# starting the ASR model training part
python3 ASR/src/model.py --pretrained-processor-path=$processor_path --model-config-path=$asr_path/exp/train_model.conf --train-csv-path="$asr_path/InputData/train/$train_csv_filename" \
                     --resampling-frequency=$resamping_frequency --test-csv-path="$asr_path/InputData/test/$test_csv_filename" --save-model-path=$model_save_path

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Now will start with the testing part of the model
