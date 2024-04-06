# The following script is used to validate the accuracy of model by testing it on testing audio dataset
# and preapring the validation graph 

# The following script is intended to use large 
# Wav2Vec 2.0 XLSR model in hindi language
# NOTE: The following libraries must be present in the python environemnt in order to run the following script

import spacy,torch, torchaudio, re
from gensim.models import FastText
import argparse, csv, os
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_metric

parser=argparse.ArgumentParser(description="",
                               epilog="",
                               formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--test-data-path", type=str, default="", required=True,
                    help="Path for the testing dataset of audio files")

parser.add_argument("--mapping-file", type=str,default="", required=True,
                    help="")

parser.add_argument("--sampling-rate", type=int, default="", required=True,
                    help="A rate at which the audio will be sampled ")

parser.add_argument("--reduce-dimension", type=int, default="", required=True,
                    help="The following argument specifies the dimension by which the"
                    "output tensor will be reduced")

# args=parser.parse_args()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# initializing all the parameters
asr_model_path="/home/sbt/HuggingFaceModels/build6/checkpoint-10000"
FastText_model_path="/home/sbt/HuggingFaceModels/build6/FastTextModel"
base_path="/home/samarth/testing/extra_testing/local_salesphony_testing/salesphony_testing/Model"
test_data_path=f"{base_path}/db/asr/general/train/audio" # the test path consisting of all the audios 
map_file=f"{base_path}/ASR/InputData/train/train_model_data.csv" # following is the mapping csv file to map the audio with corresponding transcripts
audio_sample_rate=16000 # sample rate for all audios
dimension=-1 # to be reduced of the final tensor
sentence_corpus=f"{base_path}/sentence.txt" # following is the unique vocabulary of words
f = open(f'{base_path}/ASR/exp/results.csv', 'w')
writer=csv.writer(f)

# NOTE: The column name names are aslo hardcoded in the following script 
# path: this column corresponds to the audio file ids (For e.g. 651bdd28d4c1b488bd04381f.wav)
# sentence: these are the corresponding transcript of the audio file of corresponding id

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“]'
# test_dataset = load_dataset("common_voice", "hi", split="test")
processor = Wav2Vec2Processor.from_pretrained("theainerd/Wav2Vec2-large-xlsr-hindi")
model = Wav2Vec2ForCTC.from_pretrained("theainerd/Wav2Vec2-large-xlsr-hindi")
chunk_size=1024
SAMPLE_RATE=16000

resampler = torchaudio.transforms.Resample(8000, 16000)
wer=load_metric("wer")

def process_audio_data(audio_data,mapping_file,dimension,sample_rate):
    """
    Following function will process the audio data and 
    given the prediction from pretrained HuggingFace Model
    NOTE: The mapping file will be a csv file with following features
    1) audio_id: for mapping each audio its corresponding text
    2) text: the text of the corresponding file
    3) finaltext: the final text will be corrected spelling of words in predicted transcript
    """
    
    open_mapping_file=pd.read_csv(f"{mapping_file}")
    open_audio_dir=os.listdir(audio_data)
    if len(open_mapping_file) != len(open_audio_dir):
        print("Warning: Not all data is present in either of audio data or mapping csv file")
    for audio_id in open_mapping_file["audio_path"][:100]:
        audio_id_index=open_mapping_file["audio_path"].to_list().index(audio_id)
        sentence=open_mapping_file["transcript"].to_list()[audio_id_index]
        small_sentence = re.sub(chars_to_ignore_regex, '', sentence).lower()
        # converting audio to array and resampling
        audio_array=speech_to_array(audio_path=audio_id)    
        speech_array=audio_array.get("AudioArray")
        # test the model
        model_test=build_model(audio_array=speech_array,audio_sample_rate=sample_rate,sentence=small_sentence,dimension=dimension)
        # get_final_transcript=final_spel_check(fast_text_model=fasttext_model.get("FastTextModel"),predicted_text=' '.join(model_test.get("Prediction")),word_dictionary=fasttext_model.get("Dictionary"),sentence=small_sentence)
        print("The final transcript is [ {} ]".format(model_test.get("FinalOutput")))
    print("All done")   
        
def build_fasttext_model(vocab):
    """
    The following function is used to build the fasttext model 
    for correcting the spellings in the predicted sentence
    """
    # Tokenize the sentences using spaCy
    processed_sentences = [[token.text for token in nlp(sentence.replace("\n",""))] for sentence in vocab]
    # Build a dictionary of words from the training dataset
    word_dictionary = set(word for sentence in processed_sentences for word in sentence)
    if len(os.listdir(FastText_model_path)) == 0:
        text_model = FastText(window=50)
        # Build FastText model
        text_model.build_vocab(corpus_iterable=processed_sentences)
        text_model.train(corpus_iterable=processed_sentences, total_examples=len(processed_sentences), epochs=10)
        text_model.save(f"{FastText_model_path}/fast_text_model")
        return {"FastTextModel":text_model,"Dictionary":word_dictionary}
    else:
        text_model=FastText.load(f"{FastText_model_path}/fast_text_model")
        return {"FastTextModel":text_model,"Dictionary":word_dictionary}

def speech_to_array(audio_path):
    """
    Following function takes in batch which is our test dataset
    and convert the audios into a numpy array
    """
    
    speech_array, sampling_rate = torchaudio.load(f"{audio_path}")
    resampled_array = resampler(speech_array).squeeze().numpy()
    return {"AudioArray":resampled_array}

def build_model(audio_array,audio_sample_rate,sentence,dimension):
    """
    The following function is used to prepare the ASR large Wav2Vec XLSR model
    a) test_data_path: path for the testig audio files
    b) audio_sample: the sampling rate of the audios
    c) dimension: the dimension by which the output tensor will be reduced 
    """
    inputs=processor(audio_array, sampling_rate=audio_sample_rate, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits=model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    predicted_ids=torch.argmax(logits, dim=dimension)
    prediction=processor.batch_decode(predicted_ids)
    print("Prediction:", prediction)
    print("Reference:",[sentence])
    print("WER: {:2f}".format(100 * wer.compute(predictions=[prediction], references=[sentence])))
    data = [sentence, prediction,format(100 * wer.compute(predictions=[prediction], references=[sentence]))]
    writer.writerow(data)
    return {"FinalOutput":prediction}
    
# Function to find the nearest match for a given word in the dictionary
def find_nearest_match(fast_text_model,word, dictionary):
    if word in dictionary:
        return word
    # If the word is not in the dictionary, find the most similar word
    similar_words = fast_text_model.wv.most_similar(word, topn=1)
    return similar_words[0][0] if similar_words else word

# def final_spel_check(fast_text_model,predicted_text,word_dictionary,sentence):
#     """
#     The following code is used to check the spelling 
#     of words in the final predicted transcript of the model and returns the final output
#     """
#     # Example: Replace suspicious words in the model's output
#     output_tokens = [token.text for token in nlp(predicted_text)]
#     corrected_tokens = [find_nearest_match(fast_text_model,token, word_dictionary) for token in output_tokens]
#     corrected_text = ' '.join(corrected_tokens)
#     print("Prediction:", predicted_text)
#     print("Reference:",[sentence])
#     print("WER: {:2f}".format(100 * wer.compute(predictions=[corrected_text], references=[sentence])))
#     data = [sentence, predicted_text,corrected_text ,format(100 * wer.compute(predictions=[corrected_text], references=[sentence]))]
#     writer.writerow(data)
#     return{"CorrectedText":corrected_text}

if "__main__" == __name__:
    # open_vocabulary=open(f"{sentence_corpus}").readlines()
    # fasttext_model=build_fasttext_model(vocab=open_vocabulary)
    audio_data_path="/home/samarth/testing/extra_testing/local_salesphony_testing/salesphony_testing/Model/db/asr/general/train/test"
    # testing_audio_chunks(audio_data_path,chunk_size,SAMPLE_RATE)
    process_audio_data(audio_data=test_data_path,mapping_file=map_file,dimension=dimension,sample_rate=audio_sample_rate)            