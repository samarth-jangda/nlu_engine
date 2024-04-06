#!/usr/bin/env python3

import asyncio
import json
import websockets
import sys
import wave

transcript=[]
async def run_test(uri):
    async with websockets.connect(uri) as websocket:

        wf = wave.open(sys.argv[1], "rb")
        print(wf.getframerate())
        words=["हाँ","ना","नहीं"]
        # await websocket.send('{ "config" : { "sample_rate" : %d } }' % (wf.getframerate()))
        # await websocket.send('{"config":{ "sample_rate" : 8000, "words":["हाँ","ना","नहीं"]} }')
        buffer_size = int(wf.getframerate() * 0.20) # 0.19 seconds of audio
        while True:
            data = wf.readframes(buffer_size)

            if len(data) == 0:
                break

            await websocket.send(data)
            res=await websocket.recv()
            if 'text' in res:
                json_value=json.loads(res)
                print(json_value.get("text"))
                transcript.append(json_value.get("text"))
        await websocket.send('{"eof" : 1}')
        
        final_res=await websocket.recv()
        if 'text' in final_res:
            json_value=json.loads(final_res)
            transcript.append(json_value.get("text"))
            print(json_value.get("text"))
            
        # write data to a text file
        file_path="/home/samarthjangda/testing/salesphony_testing/prediction.txt"
        transcript.pop(0)
        with open(file_path, 'w') as pred_file:
            for data in transcript:
                pred_file.writelines(f"{data}\n")
asyncio.run(run_test('ws://localhost:2700'))
