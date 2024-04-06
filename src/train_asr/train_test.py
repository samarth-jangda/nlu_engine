# The following script is used to divide the data into train and testing audio data
# The audio dataset will be the mix of general and domian audio as build by data building recipe

import os, argparse,sys
import random

parser=argparse.ArgumentParser(description="The following arguments are used to fetch the data files to split under test and train",
                               epilog="python3 --audio-data-path=<path of original audio wav and transcription files> --training-perenctage=<70> --saving-path=<path to save splitted data>",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--audio-data-path", type=str, default="", required=True,
                    help="Specify the audio data path for splitting")

parser.add_argument("--training-percentage", type=str, default="", required=True,
                    help="Specify the percentage of split")

parser.add_argument("--saving-path", type=str, default="", required=True,
                    help="Specify where to svae the training and testing data")

args=parser.parse_args()

def test_train_split(audio_path,split_percentage,saving_audio_path):
    """
    The following function is used to call the audio dataset and split the same
    under test and train dataset where:
    1) Train Data: the train data is used for model training
    2) Test Data: the test data is used for model validation 
    """
    [os.remove(f"{saving_audio_path}/{file}") for file in os.listdir(f"{saving_audio_path}/train") if len(os.listdir(f"{saving_audio_path}/train")) == 0]
    save_training_wav_file=open(f"{saving_audio_path}/train/wav.scp", 'w')
    save_training_transcription_file=open(f"{saving_audio_path}/train/transcription.txt", 'w')
    save_testing_wav_file=open(f"{saving_audio_path}/test/wav.scp", 'w')
    save_testing_transcription_file=open(f"{saving_audio_path}/test/transcription.txt", 'w')
    open_file=open(f"{audio_path}/wav.scp", 'r').readlines()
    open_wav_file=random.sample(open_file,len(open_file))
    open_transcription_file=open(f"{audio_path}/transcription").readlines()
    print("file lengths differ {} and {}".format(len(open_wav_file),len(open_transcription_file))) if len(open_wav_file) != len(open_transcription_file) else print("file lengths are same")
    file_lengths=int((len(open_wav_file) * split_percentage)/100)
    training_file_data=open_wav_file[0:file_lengths] # training data of wav files
    print(len(training_file_data))
    [save_training_wav_file.write(data) for data in training_file_data]
    testing_file_data=open_wav_file[file_lengths:] # testing data of wav files
    [save_testing_wav_file.write(data) for data in testing_file_data]
    training_wav_ids=[file_id.split(" ")[0] for file_id in training_file_data]
    testing_wav_ids=[file_id.split(" ")[0] for file_id in testing_file_data]
    training_transcription_data=[save_training_transcription_file.write(text) for wav_id in training_wav_ids for text in open_transcription_file if wav_id in text] # training data of transcription files
    testing_transcription_data=[save_testing_transcription_file.write(text) for wav_id in testing_wav_ids for text in open_transcription_file if wav_id in text] # testing data of transcription files
    return (print("Completed with splitting audio data in training and testing"))    
if "__main__" == __name__:
    test_train_split(args.audio_data_path,int(args.training_percentage),args.saving_path)