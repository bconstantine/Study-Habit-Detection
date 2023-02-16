#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from multiprocessing.sharedctypes import Value
from typing import Union
from pydantic import BaseModel
from uuid import uuid4
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *
import os
import random

#pydantic BaseModel Class (Employee Data)
class HistoryRecord(BaseModel):
    date:str = ""
    good: Union[float, None] = None
    cheek: Union[float, None] = None
    forehead: Union[float, None] = None

dataName = 'HistoryRecord.json'
HistoryData = []

def readJson():
    #load json data
    global HistoryData
    if os.path.exists(dataName):
        with open(dataName, "r") as file:
            HistoryData = json.load(file)
def printAllString():
    retStr = "===== history data =====\n"
    for i in HistoryData:
        curDate = i["date"]
        curGood = i["good"]
        curCheek = i["cheek"]
        curForehead = i["forehead"]
        retStr+=f"Session finish time: {curDate}\n"
        retStr += f'Good posture time: {round(curGood,2)}\n'
        retStr += f'Bad posture，hand on cheek time: {round(curCheek,2)}\n'
        retStr += f'Bad posture，hand on forehead posture time: {round(curForehead,2)}\n'
        retStr += "=====              =====\n"
    return retStr

def evalLastSession():
    readJson()
    retStr = ""
    if len(HistoryData) == 0:
        retStr += "Hmm... You haven't done any completed session, please study more!\n"
    else:
        retStr += "Hmm.. Let's see...\n\n"
        data = HistoryData[-1]
        maxAll = max(data['good'], max(data['cheek'],data['forehead']))
        if maxAll == data['good']:
            retStr += "From what I have seen, your longest posture time is good posture, keep it up!\n"
        elif maxAll == data['cheek']:
            retStr += "Whoa! Your longest posture time is bad posture, hand on cheek. Make sure to fix this on next session!\n"
        else:
            retStr += "Something's not right! Your longest posture time is bad posture, hand on forehead. Make sure to fix this on next session!\n"
        return retStr 
def writeToJson():
    with open(dataName, "w") as f: 
        json.dump(HistoryData, f, indent = 4)

def clearJson():
    HistoryData.clear()
    with open(dataName, "w") as f: 
        json.dump(HistoryData, f, indent = 4)

def addEntry(date:str = "", good:float = 0, cheek:float = 0, forehead:float = 0):
    data_id = uuid4().hex
    history_dict = HistoryRecord().dict()
    history_dict.update({"id":data_id})
    history_dict["date"] = date
    history_dict["good"] = good
    history_dict["cheek"] = cheek
    history_dict["forehead"] = forehead
    
    HistoryData.append(history_dict)
    writeToJson()
    
load_dotenv() # Load your local environment variables
CHANNEL_TOKEN = os.environ.get('LINE_TOKEN')
CHANNEL_SECRET = os.getenv('LINE_SECRET')
app = FastAPI()

My_LineBotAPI = LineBotApi(CHANNEL_TOKEN) # Connect Your API to Line Developer API by Token
handler = WebhookHandler(CHANNEL_SECRET) # Event handler connect to Line Bot by Secret key
CHANNEL_ID = os.getenv('LINE_UID') # For any message pushing to or pulling from Line Bot using this ID
#My_LineBotAPI.push_message(CHANNEL_ID, TextSendMessage(text='Welcome to the app !')) # Push a testing message
my_event = ['#help', '#report', '#clear', '#eval']
# Create my emoji list
my_emoji = [
    [{'index':27, 'productId':'5ac1bfd5040ab15980c9b435', 'emojiId':'005'}],
    [{'index':27, 'productId':'5ac1bfd5040ab15980c9b435', 'emojiId':'019'}],
    [{'index':27, 'productId':'5ac1bfd5040ab15980c9b435', 'emojiId':'096'}]
]

@app.post('/')
async def callback(request: Request):
    #print('enter here 2')
    body = await request.body() # Get request
    signature = request.headers.get('X-Line-Signature', '') # Get message signature from Line Server
    try:
        handler.handle(body.decode('utf-8'), signature) # Handler handle any message from LineBot and 
    except InvalidSignatureError:
        raise HTTPException(404, detail='LineBot Handle Body Error !')
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_textmessage(event):
    #print('enter here0')
    receive_message = str(event.message.text).split(' ')
    formatVerified = False
    # Get first splitted message as command
    case_ = receive_message[0].lower().strip()
    #print('enter here1')
    if len(receive_message) == 1 and case_ in my_event:
        formatVerified = True
    if(formatVerified):
        if my_event[0] == case_:
            command_description = '$ Commands:\n            Welcome! I can assist you in getting the record of your study habit!\n\n            report: #report \n\t--> Show all of your recorded study sessions\n\
            clear: #clear --> clear all sessions data\n\
            eval: #eval --> I will review your last finished session!\n\
            or send me any sticker to get a sticker reply!\n\n\n'
            
            My_LineBotAPI.reply_message(
                event.reply_token,
                TextSendMessage(
                    text=command_description,
                    emojis=[
                        {
                            'index':0,
                            'productId':'5ac21a18040ab15980c9b43e',
                            'emojiId':'110'
                        }
                    ]
                )
            )
        elif my_event[1] == case_:
            #report
            readJson()
            res = printAllString()
            My_LineBotAPI.reply_message(
                event.reply_token,
                TextSendMessage(
                    text=res,
                )
            )
        elif my_event[2] == case_:
            #clear
            clearJson()
            My_LineBotAPI.reply_message(
                event.reply_token,
                TextSendMessage(
                    text='Cleared!',
                )
            )
        else: 
            #eval
            res = evalLastSession()
            My_LineBotAPI.reply_message(
                event.reply_token,
                TextSendMessage(
                    text=res,
                )
            )
    else:
        My_LineBotAPI.reply_message(
            event.reply_token,
            TextSendMessage(
                text='$ Welcome to Study habit database ! Your command is unrecognized! Enter "#help" for commands !',
                emojis=[
                    {
                        'index':0,
                        'productId':'5ac2213e040ab15980c9b447',
                        'emojiId':'035'
                    }
                ]
            )
        )
# Line Sticker Class
class My_Sticker:
    def __init__(self, p_id: str, s_id: str):
        self.type = 'sticker'
        self.packageID = p_id
        self.stickerID = s_id
'''
See more about Line Sticker, references below
> Line Developer Message API, https://developers.line.biz/en/reference/messaging-api/#sticker-message
> Line Bot Free Stickers, https://developers.line.biz/en/docs/messaging-api/sticker-list/
'''
# Add stickers into my_sticker list
my_sticker = [My_Sticker(p_id='446', s_id='1995'), My_Sticker(p_id='446', s_id='2012'),
     My_Sticker(p_id='446', s_id='2024'), My_Sticker(p_id='446', s_id='2027'),
     My_Sticker(p_id='789', s_id='10857'), My_Sticker(p_id='789', s_id='10877'),
     My_Sticker(p_id='789', s_id='10881'), My_Sticker(p_id='789', s_id='10885'),
     ]
# Line Sticker Event
@handler.add(MessageEvent, message=StickerMessage)
def handle_sticker(event):
    # Random choice a sticker from my_sticker list
    ran_sticker = random.choice(my_sticker)
    # Reply Sticker Message
    My_LineBotAPI.reply_message(
        event.reply_token,
        StickerSendMessage(
            package_id= ran_sticker.packageID,
            sticker_id= ran_sticker.stickerID
        )
    )
if __name__=='__main__':
    print('enter here-2')
    import uvicorn
    uvicorn.run(app='main:app', reload=True, host='0.0.0.0', port=58821)


# In[ ]:





# In[ ]:




