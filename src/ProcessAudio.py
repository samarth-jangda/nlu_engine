# The following script is used to process the audio frames by:
# 1) Removing Noise: Every live stream signal has a noise present in it which can be removed
# 2) Removing Silence: During merging the audio when generating final transcipt un necessary silence will be removed

import librosa
import numpy as np
from scipy.interpolate import interp1d

# initializing parameters
silence_threshold=-2.0
process_audio=True
sample_rate=8000
mfcc_number=100
hop_size=512
bitrate=np.int16
attenuation_factor=0.5

def process_audio_frames(audio_frame):
    """
    The following function is used to pass the audio buffer array
    to noise removal and silence detection 
    """
    audio_array=np.frombuffer(audio_frame,dtype=bitrate)
    float_audio=audio_array.astype(np.float32)/np.iinfo(bitrate).max
    filter_audio=noise_subtraction(float_audio,hop_size,attenuation_factor)
    filtered_audio=(filter_audio*np.iinfo(bitrate).max).astype(bitrate)
    silence=detect_silence(filtered_audio)
    return silence

def noise_subtraction(audio_buffer,hop_length,attn_factor):
    """
    Following function is used to subtract the noise from an original audio
    streamed buffer using spectral subtraction
    """
    stft=librosa.stft(audio_buffer,n_fft=len(audio_buffer),hop_length=int(len(audio_buffer)/2))
    audio_magnitude=np.abs(stft)
    noisy_audio_profile=np.mean(audio_magnitude,axis=1)
    noisy_audio_expanded=np.expand_dims(noisy_audio_profile, axis=1)
    filtered_audio_stft=stft-attn_factor*noisy_audio_expanded
    return filtered_audio_stft
    
def detect_silence(float_audio):
    """
    The following function creates a logic of
    1) Detecting silence in a live audio frame as:
        speech: If speech then combine with previous data
        silence If silence then process final and clear the list
    2) Process the partials and final from ASR model    
    """
    float_audio = float_audio.astype(np.float32)/np.iinfo(bitrate).max
    mfccs = librosa.feature.mfcc(y=float_audio, sr=sample_rate, n_mfcc=mfcc_number,n_fft=len(float_audio))
    energy = np.mean(mfccs)
    print(f"Energy:{energy}")
    # Check if energy is below the threshold
    if energy <= -2.30:
        return True
    else:
        return False
