# The following script is the model_server which is used to:
# 1) Live Streams: Fetch live audio streams from twilio
# 2) Predict Transcript: Predict the transcript using HuggingFace Wav2Vec2.0 model of the live chunks
# 3) Predict entities: Predict the entities from the live transcript of chunks
# 4) Report Generation: Finally generating the report on the basis of transcript and entities

import os
import asyncio
import websockets
import logging
import json
from vox import ModelRecognizer
import concurrent.futures
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

word_list=[]
def fetch_transcript(response,message):
    """
    The following function is used to get the partial and
    final transcript of the audio frame from ASR model 
    """
    if response.get("message")=='{"eof":1}':
        return {"FinalTranscript":response.get("Transcript")},False
    elif response.get("message")=="silence":
        return response,False
    else:
        return {"PartialTranscript":response.get("Transcript")},False

async def handle_websocket(websocket):
    global args
    global pool
    audio_path="/home/samarthjangda/testing/sample.wav"
    # output_file=wave.open(audio_path, 'wb')
    # output_file.setnchannels(1)  # Mono
    # output_file.setsampwidth(2)  # 16-bit
    # output_file.setframerate(8000)  # Sample rate
    
    # global predicted_chunk_transcript
    loop=asyncio.get_running_loop()
    print("WebSocket connection established")
    # Function to handle incoming WebSocket messages
    logging.info('Connection from %s', websocket.remote_address);
    words=None
    res=None
    sample_rate=None
    alternatives=None
    while True:
        message=await websocket.recv()
        if isinstance(message,str) and 'config' in message:
            jobj=json.loads(message)['config']
            logging.info("Config %s",jobj)
            if "words" in jobj:
                words=jobj["words"]
            if "sample_rate" in jobj:
                sample_rate=jobj["sample_rate"]
            if "alternatives" in jobj:
                alternatives=jobj["alternatives"]
            continue
        # if not res and type(message)==bytes:
        if words:
            res=ModelRecognizer.FetchData.load_data(message,asr_processor,asr_model,word_list)
        else:
            res=ModelRecognizer.FetchData.load_data(message,asr_processor,asr_model)
            # condition of partials and final transcript prediction
        response,stop=await loop.run_in_executor(pool,fetch_transcript,res,message)
        print(response)
        await websocket.send(str(response))
        if stop: break    
           
                    
    # output_file.close()       
            
async def start():
    global args
    global pool
    global asr_model
    global asr_processor
    
    # start logging 
    logging.basicConfig(level=logging.INFO)
    args=type('', (), {})()
    # Make sure to set the values of specific environemnt variable values
    args.interface=os.environ.get("Verbi_IP",'127.0.0.1') # ---------------------------------------> The IP address of the model server
    args.port=os.environ.get("Verbi_port",2710) # -------------------------------------------------> The port of the model server
    # args.asr_model_path=os.environ.get("Verbi_asr_model_path",'asr_model') # --------------------> The actual prepared asr model path
    # args.ner_model_path=os.environ.get("Verbi_ner_model_path",'ner_model') # --------------------> The actual prepared ner model path
    args.gmm_model_path=os.environ.get("Verbi_gmm_model_path",'gmm_model') # ----------------------> The actual prepared gmm model path
    args.fastText_model_path=os.environ.get("Verbi_fasttext_model_path",'fastText_model') # -------> The actual prepared fastText model path
    # loading all models
    asr_processor = Wav2Vec2Processor.from_pretrained("theainerd/Wav2Vec2-large-xlsr-hindi")
    asr_model = Wav2Vec2ForCTC.from_pretrained("theainerd/Wav2Vec2-large-xlsr-hindi")
    pool=concurrent.futures.ThreadPoolExecutor(os.cpu_count() or 1)
    # start serving websocket
    async with websockets.serve(handle_websocket,args.interface,args.port,ping_interval=None):
        await asyncio.Future()
        
if __name__=="__main__":
    asyncio.run(start())        