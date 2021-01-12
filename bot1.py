# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:05:57 2021

@author: useren
"""

import logging
from aiogram import Bot, Dispatcher, executor, types
import aiogram

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
    await message.reply("Hi!\nI'm EchoBot!\nPowered by aiogram.")
    
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)