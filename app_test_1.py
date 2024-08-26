import os
from dotenv import load_dotenv

import asyncio
from typing import Annotated

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatImage,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import geocoder

load_dotenv()

class AssistantFunction(agents.llm.FunctionContext):
    """This class is used to define functions that will be called by the assistant."""

    @agents.llm.ai_callable(
        description=(
            "Called when asked to evaluate something that would require vision capabilities, do not call when asked about weather"
            "for example, an image, video, or the webcam feed."
        )
    )
    async def image(
        self,
        user_msg: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The user message that triggered this function"
            ),
        ],
    ):
        print(f"Message triggering vision capabilities: {user_msg}")
        return None

    # callable that calls weather api to get weather information
    @agents.llm.ai_callable(
        description=(
            "Called when asked to tell the user information about the weather,"
            "for example, 'tell me what the weather is right now'"
        )
    )
    async def weather(
        self,
        user_msg: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The user message that triggered this function"
            ),
        ],
    ):
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)

        # get latitude and longitude of current location
        g = geocoder.ip('me')
        lat = g.latlng[0]
        long = g.latlng[1]

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": long,
            "hourly": "temperature_2m"
        }
        responses = openmeteo.weather_api(url, params=params)

        response = responses[0]
        print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
        print(f"Elevation {response.Elevation()} m asl")
        print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
        print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

        hourly_data = {"date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}
        hourly_data["temperature_2m"] = hourly_temperature_2m

        hourly_dataframe = pd.DataFrame(data = hourly_data)
        weather_info = hourly_dataframe.head(1).to_dict()
        text = f"The temperature is currently {weather_info['temperature_2m'][0]} degrees"
        print(f"Weather Data Received {hourly_dataframe}")
        
        return text
    

async def get_video_track(room: rtc.Room):
    """Get the first video track from the room. We'll use this track to process images."""

    video_track = asyncio.Future[rtc.RemoteVideoTrack]()

    for _, participant in room.remote_participants.items():
        for _, track_publication in participant.track_publications.items():
            if track_publication.track is not None and isinstance(
                track_publication.track, rtc.RemoteVideoTrack
            ):
                video_track.set_result(track_publication.track)
                print(f"Using video track {track_publication.track.sid}")
                break

    return await video_track


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"Room name: {ctx.room.name}")

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Your name is Alloy. You are a funny, witty bot that is helping a rowing team answer questions during a race." 
                    "Your interface with users will be voice and vision."
                    "Respond with short and concise answers. Avoid using unpronouncable punctuation or emojis."
                ),
            )
        ]
    )

    gpt = openai.LLM(model="gpt-4o")

    # Since OpenAI does not support streaming TTS, we'll use it with a StreamAdapter
    # to make it compatible with the VoiceAssistant
    openai_tts = tts.StreamAdapter(
        tts=openai.TTS(voice="alloy"),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
    )

    latest_image: rtc.VideoFrame | None = None

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),  # We'll use Silero's Voice Activity Detector (VAD)
        stt=deepgram.STT(),  # We'll use Deepgram's Speech To Text (STT)
        llm=gpt,
        tts=openai_tts,  # We'll use OpenAI's Text To Speech (TTS)
        fnc_ctx=AssistantFunction(),
        chat_ctx=chat_context,
    )

    chat = rtc.ChatManager(ctx.room)

    async def _answer(text: str, use_image: bool = False):
        """
        Answer the user's message with the given text and optionally the latest
        image captured from the video track.
        """
        content: list[str | ChatImage] = [text]
        if use_image and latest_image:
            content.append(ChatImage(image=latest_image))

        chat_context.messages.append(ChatMessage(role="user", content=content))

        stream = gpt.chat(chat_ctx=chat_context)
        await assistant.say(stream, allow_interruptions=True)

    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        """This event triggers whenever we get a new message from the user."""

        if msg.message:
            asyncio.create_task(_answer(msg.message, use_image=False))

    
    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        """This event triggers when an assistant's function call completes."""

        if len(called_functions) == 0:
            return
        
        print(called_functions)

        for function in called_functions:
            if function.call_info.function_info.name == 'weather':
                weather_data = function.task.result()
                print(f"\n \n {weather_data} \n \n")
        
        '''
        user_msg = called_functions[0].call_info.arguments.get("user_msg")

        if user_msg:
            asyncio.create_task(_answer(user_msg, use_image=True))
        '''
        if weather_data:
            asyncio.create_task(_answer(weather_data, use_image=False))

    assistant.start(ctx.room)

    await asyncio.sleep(1)
    await assistant.say("Hi there! How can I help?", allow_interruptions=True)

    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        video_track = await get_video_track(ctx.room)

        async for event in rtc.VideoStream(video_track):
            # We'll continually grab the latest image from the video track
            # and store it in a variable.
            latest_image = event.frame


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

