# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:05:57 2021

@author: useren
"""

import logging
from aiogram import Bot, executor, types
from aiogram.dispatcher import Dispatcher
#from aiogram.dispatcher.webhook import SendMessage
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.utils.executor import start_webhook
from aiohttp import ClientSession
import aiogram
from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton
from urllib.parse import urljoin
from PIL import Image
import io
import os


webhook_using = False
if os.name == 'posix':
    webhook_using = True
    API_TOKEN = os.getenv('API_TOKEN','123213:SDFSDGSD_ASDKKDF')
else:
    with open('API.TOKEN', 'r') as f:
        API_TOKEN = f.readline().split()[0]

#webhook setting

WEBHOOK_HOST = 'https://telegabot67.heroku.com'
WEBHOOK_PATH = '/webhook/'+API_TOKEN
WEBHOOK_URL = urljoin(WEBHOOK_HOST, WEBHOOK_PATH)
print(f'wh_url=\n{WEBHOOK_URL}, type({type(WEBHOOK_URL)}) ?\n{WEBHOOK_HOST + WEBHOOK_PATH}')


#webapp setting
if webhook_using:
    WEBAPP_HOST = '0.0.0.0'
    WEBAPP_PORT = os.getenv('PORT')
    print(WEBAPP_PORT)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
if webhook_using:
    dp.middleware.setup(LoggingMiddleware())


async def on_startup(dp):
    logging.warning('++++starting webhook')
   # await bot.delete_webhook()
    await bot.set_webhook(WEBHOOK_URL)
    
    
async def on_shutdown(dp):
    logging.warning('+++Shutting down...')
    
#    await bot.delete_webhook()
    
    await dp.storage.close()
    await dp.storage.wait_closed()
    
    logging.warning('+++Bye-bye!')
    

@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.reply(f"Hi!\nI'm EchoBot!\nos.name={os.name}")
@dp.message_handler()
async def echo(message: types.Message):
    # old style:
    # await bot.send_message(message.chat.id, message.text)
    mes_to_answ = ''
    mes_to_answ += ' date: ' + str(message.date)
    await message.answer(mes_to_answ)
if __name__ == '__main__':
    if webhook_using:
        logging.warning(f'---->trying start webhook:{WEBHOOK_PATH}, {WEBAPP_HOST}, {WEBAPP_PORT}')
        start_webhook(dispatcher=dp,
                      webhook_path=WEBHOOK_PATH,
                      on_startup=on_startup,
                      on_shutdown=on_shutdown,
                      skip_updates=True,
                      host=WEBAPP_HOST,
                      port=WEBAPP_PORT)
    else:
        executor.start_polling(dp, skip_updates=True)
        