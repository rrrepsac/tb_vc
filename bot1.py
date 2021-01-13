# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:05:57 2021

@author: useren
"""

import logging
from aiogram import Bot, Dispatcher, executor, types
import aiogram
from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton
    
#print(logging.__version__, aiogram.__version__)
#assert False

API_TOKEN = '1519854172:AAFHk6QYK7ak_YWyMb8uiQtzK83kUSG5VJg'

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


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
    await message.reply("Hi!\nI'm EchoBot!\nPowered by aiogram.",\
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

    await message.answer(message.text)
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)