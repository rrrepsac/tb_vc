import os
import logging
import asyncio
from aiogram import Bot, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.dispatcher import Dispatcher
from aiogram.utils.executor import start_webhook
from inference import JohnsonMultiStyleNet, make_style
from PIL import Image
import io
import numpy as np

#KMP_DUPLICATE_LIB_OK=TRUE
#os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'


style_num = 11
style_model = JohnsonMultiStyleNet(style_num)
style_model.eval()
style_names = ['candy', 'cezanne', 'mosaic', 'picasso', 'rain-princess', 
               'stary-night', 'undie', 'waterfall', 'kandinsky', 'mona lisa',
               'peppa']

BOT_TOKEN = os.getenv('API_TOKEN')
if not BOT_TOKEN:
    print('You have forgot to set BOT_TOKEN')
    with open('API.TOKEN', 'r') as f:
        BOT_TOKEN = f.readline().split()[0]
        logging.warning(f'token = {BOT_TOKEN}')
    if not BOT_TOKEN:
        quit()

HEROKU_APP_NAME = 'telegabot67'  # os.getenv('HEROKU_APP_NAME')


# webhook settings
WEBHOOK_HOST = f'https://{HEROKU_APP_NAME}.herokuapp.com'
WEBHOOK_PATH = f'/webhook/{BOT_TOKEN}'
WEBHOOK_URL = f'{WEBHOOK_HOST}{WEBHOOK_PATH}'

# webserver settings
WEBAPP_HOST = '0.0.0.0'
WEBAPP_PORT = int(os.getenv('PORT'))


bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.reply(
f"Hi!\nI'm MultiStyle Bot!\nI can transfer {style_model.get_style_number()} styles.\n\
If you send me any message, I'll style test.jpg with random style and send it to you.\n\
If I get a photo, I send back random styled photo.\n\
You can specify the number of style from [0 to {style_model.get_style_number()-1}]\n\
{[(i, x) for (i, x) in enumerate(style_names)]}")

@dp.message_handler()
async def echo(message: types.Message):
    img = Image.open('test.jpg')
    fp = io.BytesIO()
    style_num = np.random.randint(style_model.get_style_number())
    await message.answer(f'Styling like {style_names[style_num]}')
    styled, style_num = make_style(img, style_model, style_num)
    
    Image.fromarray(styled).save(fp, 'JPEG')
    
    await bot.send_photo(message.from_user.id, fp.getvalue(),
                         reply_to_message_id=message.message_id)

def get_first_num(string=None, module=None, default=None):
    if string and type(string) is str:
        digits = [word for word in string.split() if word.isdigit()]
        if digits:
            num = int(digits[0])
        if module:
            num = num % module
            return num
    return default
@dp.message_handler(content_types=['photo'])
async def photo_reply(message: types.Message):
    fpin = io.BytesIO()
    fpout = io.BytesIO()
    style_num = np.random.randint(style_model.get_style_number())
    logging.warning(f'rand {style_num}')
    logging.warning(f'mestxt={message.text}, caption={message.caption}')
    style_num = get_first_num(message.text, style_model.get_style_number(), style_num)
    style_num = get_first_num(message.caption, style_model.get_style_number(), style_num)
    await message.answer(f'Your photo will be styled like {style_names[style_num]}')
    await message.photo[-1].download(fpin)
    img = Image.open(fpin)
    logging.warning(f'call {style_num}')

    styled, style_num = make_style(img, style_model, style_num)
    logging.warning(f'get {style_num}')

    Image.fromarray(styled).save(fpout, 'JPEG')
    
    #fid=message.photo[-1].file_id
    #print(fid)
    await bot.send_photo(message.from_user.id, fpout.getvalue(),
                         reply_to_message_id=message.message_id)

async def on_startup(dp):
    logging.warning(
        'Starting connection. ')
    await bot.set_webhook(WEBHOOK_URL,drop_pending_updates=True)


async def on_shutdown(dp):
    logging.warning('Bye! Shutting down webhook connection')


def main():
    logging.basicConfig(level=logging.INFO)
    start_webhook(
        dispatcher=dp,
        webhook_path=WEBHOOK_PATH,
        skip_updates=True,
        on_startup=on_startup,
        host=WEBAPP_HOST,
        port=WEBAPP_PORT,
    )
    

#def set_hook():
#    bot = Bot(token=BOT_TOKEN)

#     async def hook_set():
#        if not HEROKU_APP_NAME:
#            print('You have forgot to set HEROKU_APP_NAME')
#            quit()
#        await bot.set_webhook(WEBHOOK_URL)
#        print(await bot.get_webhook_info())
    

#    asyncio.run(hook_set())
#    bot.close()


#def start():
    #main()
if __name__ == '__main__':
    main()