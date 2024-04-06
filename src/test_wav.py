#!/usr/bin/env python3

import asyncio
import websockets
import sys
import wave

async def run_test(uri):
    async with websockets.connect(uri) as websocket:
        wf = wave.open(sys.argv[1], "rb")
        words=["हाँ","ना","नहीं"]
        # await websocket.send('{ "config" : { "sample_rate" : %d } }' % (wf.getframerate()))
        # await websocket.send('{"config":{ "sample_rate" : 8000, "words":["हाँ","ना","नहीं"]} }')
        buffer_size = int(wf.getframerate() * 0.25) # 0.25 seconds of audio
        while True:
            data = wf.readframes(buffer_size)

            if len(data) == 0:
                break

            await websocket.send(data)
            print (await websocket.recv())

        await websocket.send('{"eof" : 1}')
        print (await websocket.recv())

asyncio.run(run_test('ws://localhost:2700'))
