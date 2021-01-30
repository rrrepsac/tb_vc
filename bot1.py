# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:05:57 2021

@author: useren
"""

import logging
from aiogram import Bot, executor, types
from aiogram.dispatcher import Dispatcher
from aiogram.dispatcher.webhook import SendMessage
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.utils.executor import start_webhook
import aiogram
from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton
from PIL import Image
import io
import os

#print(logging.__version__, aiogram.__version__)
#assert False

API_TOKEN = '1519854172:AAFHk6QYK7ak_YWyMb8uiQtzK83kUSG5VJg'

#webhook setting

WEBHOOK_HOST = ''
WEBHOOK_PATH = ''#'webhook/'+API_TOKEN
WEBHOOK_URL = WEBHOOK_HOST + WEBHOOK_PATH


#webapp setting

WEBAPP_HOST = 'localhost'
WEBAPP_PORT = 3001

print(os.name)
webhook_using = False


# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)
if webhook_using:
    dp.middleware.setup(LoggingMiddleWare())


async def on_startup(dp):
    await bot.set_webhook(WEBHOOK_URL)
    
    
async def on_shutdown(dp):
    logging.warning('Shutting down...')
    
    await bot.delete_webhook()
    
    await dp.storage.close()
    await dp.storage.wait_closed()
    
    logging.warning('Bye-bye!')
    

@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    """
    This handler will be called when user sends `/start` or `/help` command
    """
    button1 = KeyboardButton('button1')
    button2 = KeyboardButton('button2')
    button3 = KeyboardButton('button1')
    button4 = KeyboardButton('button2')
    keyboard1 = ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard1.add(button1)
    keyboard1.add(button2)
    keyboard1.row(button3, button4)
    await message.reply(f"Hi!\nI'm EchoBot!\nPowered by aiogram.os.name={os.name}",\
                        reply_markup=keyboard1)
@dp.message_handler(commands=['1'])
async def comm1(message: types.Message):
    # old style:
    # await bot.send_message(message.chat.id, message.text)

    buttons = [['1butt1'],['1butt2']]
    keyboard1 = aiogram.types.inline_keyboard.InlineKeyboardMarkup(buttons)
    await message.answer(message.text, reply_markup=(keyboard1))
@dp.message_handler(commands=['2'])
async def comm2(message: types.Message):
    # old style:
    # await bot.send_message(message.chat.id, message.text)
    buttons = [['2butt1'],['2butt2']]
    keyboard1 = aiogram.types.inline_keyboard.InlineKeyboardMarkup(buttons)

    await message.answer(message.text, reply_markup=keyboard1)
@dp.message_handler(commands=['5'])
async def comm5(message: types.Message):
    # old style:
    # await bot.send_message(message.chat.id, message.text)
    buttons = ['5butt1','5butt2']
    keyboard1 = aiogram.types.inline_keyboard.InlineKeyboardMarkup(buttons)

    await message.answer(message.text, reply_markup=keyboard1)
@dp.message_handler(commands=['6'])
async def comm6(message: types.Message):
    # old style:
    # await bot.send_message(message.chat.id, message.text)
    buttons = ['6butt1','6butt2']
    keyboard1 = aiogram.types.inline_keyboard.InlineKeyboardMarkup(buttons)

    await message.answer(message.text, reply_markup=(keyboard1))
@dp.message_handler(commands=['3'])
async def comm3(message: types.Message):
    # old style:
    # await bot.send_message(message.chat.id, message.text)

    await message.answer('comm3 answer!')
@dp.message_handler()
async def echo(message: types.Message):
    # old style:
    # await bot.send_message(message.chat.id, message.text)
    mes_to_answ = ''
    if message.text is not None:
        mes_to_answ += message.text
    else:
        mes_to_answ += 'not text_mess '
    if message.photo is not None:
        mes_to_answ += str(len(message.photo))
    #mes_to_answ += message.from.first_name
    mes_to_answ += '_' + str(message.date)
    await message.answer(mes_to_answ)
@dp.message_handler(content_types=['photo'])
async def voice_reply(message: types.Message):
    await message.photo[-1].download('test.jpg')
    print(fid:=message.photo[-1].file_id)
    await message.answer(f'Get photo! {fid}')
    #await message.reply_photo(photo=
    #'AgACAgIAAxkBAAOrYBSKWkV8E2pfrUSBDA_M66QVEIYAAiCxMRtqrqhICQ9Xs-KAc8877ReYLgADAQADAgADeAADRE4GAAEeBA')
    img = Image.open('220_facades.png')
    print('PIL opened')
    bimg = io.BytesIO()
    bimg.name = 'f.png'
    print(f'.{(bimg)}')
    img.save(bimg, 'png')
    img.save('3.png', 'png')
    print(f'.{(bimg)}')
    bimg.seek(0)
    await message.reply_photo(photo=bimg)
    print('.')
if __name__ == '__main__':
    if webhook_using:
        start_webhook(dp, WEBHOOK_PATH, on_startup=on_startup,
                      on_shutdown=on_shutdown, skip_updates=False,
                      host=WEBAPP_HOST, port=WEBAPP_PORT)
    else:
        executor.start_polling(dp, skip_updates=True)
        