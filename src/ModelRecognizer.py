# The following script is used to process the audio buffers to :
# 1) GMM Model: The GMM model is used to find if buffer is speech or silence
# 2) ASR Model: The ASR Model is used to predict the transcript of speech buffer
# 3) FastText Model: The fastText model is used to correct the word from the transcript
# NOTE: Data in Queue will be only added if NO SILENCE is detected
# NOTE: When testing from Twilio make sure to set dtype as int8 or int8

import torch
import librosa
import torchaudio
from . import ProcessAudio
import numpy as np

# initializing some parameters
UPSAMPLE_RATE=16000
resampler=torchaudio.transforms.Resample(8000,UPSAMPLE_RATE)

class FetchData:
    data_variable=()
    @classmethod
    def load_data(cls,*args):
        cls.data_variable=args    
        audio_buffer=cls.data_variable[0]
        asr_processor=cls.data_variable[1]
        asr_model=cls.data_variable[2]
        # get_final=cls.data_variable[3]
        output=process_audio(audio_buffer,asr_processor,asr_model)
        return output

# Building a Queue 
class Queue:
    _items: list=[]
    @property
    def items(self):
        return self._items
    
    def enqueue(self,item):
        self._items.append(item)
    
    def dequeue(self):
        return self._items.pop(0) if not self.is_empty() else None
    
    def is_empty(self):
        return len(self.items)==0

def get_transcript(audio_buffer,process_partial,asr_processor,asr_model):
    """
    Following function 
    will:
    1) Send the audio buffer to ASR model 
    2) And fetch transcript 
    """
    if process_partial:
        chunk_array=np.frombuffer(audio_buffer,dtype=np.int8) 
        upsample_array=upsample_audio(chunk_array)
        get_partial=asr_prediction(upsample_array.get("UpsampledChunk"),UPSAMPLE_RATE,asr_processor,asr_model,-1)
        # add data to Queue
        return get_partial
    else:
        # process the final transcript
        chunk_array=np.frombuffer(audio_buffer,dtype=np.int8)
        upsample_array=upsample_audio(chunk_array)
        final_transcript=asr_prediction(upsample_array.get("UpsampledChunk"),UPSAMPLE_RATE,asr_processor,asr_model,-1)
        return final_transcript
        
def process_audio(audio_buffer,asr_processor,asr_model):
    """
    The following function is used to pass the buffer to
    1) DetectSilence: To detect silence in a live audio buffer using ffmpeg PIPELINE
    2) UpsampleAudio: Upsample the audio buffer if it is a speech
    3) ASRPrediction:: To get the Partial and Final Transcript
    """
    check_silence=ProcessAudio.process_audio_frames(audio_buffer)
    Q=Queue()
    data=Q.dequeue()
    if audio_buffer == '{"eof" : 1}':
        transcript=get_transcript(data,False,asr_processor,asr_model)
        return {"Transcript":transcript,"message":'{"eof":1}'}
    
    elif check_silence == True:
        # check if Queue has data
        if data is None:
            # audio has silence
            return {"message":"silence"}
        else:
            # remove_silence=ProcessAudio.remove_silence(data)
            transcript=get_transcript(data,False,asr_processor,asr_model)
            return {"Transcript":transcript,"message":'{"eof":1}'}
        
    else:
        # Check if Queue has data
        if data is None:
            audio_array=np.frombuffer(audio_buffer,dtype=np.int8)
            float_audio = audio_array.astype(np.float32)/np.iinfo(np.int8).max
            unnoisy_audio_buffer=ProcessAudio.noise_subtraction(float_audio,int(len(float_audio)/2),0.5)
            filtered_audio=(unnoisy_audio_buffer*np.iinfo(np.int8).max).astype(np.int8)
            filtered_audio_bytes = filtered_audio.tobytes()
            transcript=get_transcript(filtered_audio_bytes,True,asr_processor,asr_model)
            Q.enqueue(audio_buffer)
            return {"Transcript":transcript}
        else:
            max_length = max(len(data), len(audio_buffer))
            data=data.ljust(max_length,b'\x00')
            audio_buffer=audio_buffer.ljust(max_length,b'\x00')
            merged_data=data+audio_buffer
            # unsilenced_buffer=ProcessAudio.remove_silence(merged_data)
            transcript=get_transcript(merged_data,True,asr_processor,asr_model)
            Q.enqueue(merged_data)
            return {"Transcript":transcript}
        
def upsample_audio(audio_chunk):
    """
    The following function is used to upsample the audio present
    in speech pool
    """
    audio_tensor=torch.from_numpy(audio_chunk).float()
    upsampled_tensor=resampler(audio_tensor)
    upsampled_array=upsampled_tensor.numpy()
    # upsampled_chunk=signal.resample(chunk_array,chunk_size*sample_rate//old_sample_rate)
    return {"UpsampledChunk":upsampled_array}

def asr_prediction(resulted_upsample_chunk,UPSAMPLE_RATE,model_processor,asr_model,dimension):
    """
    The following function is used to predict the transcript of data in the
    pool. The data will be in following manner
    Pred1: { partial1 }
    Pred2: { partial1 partial2 }
    .....................................................
    PredN: { partial1 partial2 partial3 ........partialN }
    """
    inputs=model_processor(resulted_upsample_chunk, sampling_rate=UPSAMPLE_RATE, return_tensors="pt", padding=False)
    with torch.no_grad():
        logits=asr_model(inputs.input_values,attention_mask=inputs.attention_mask).logits
    predicted_ids=torch.argmax(logits, dim=dimension)
    output=model_processor.batch_decode(predicted_ids)
    # partials["Partials"]=output
    return output