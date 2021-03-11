from telegram.ext import Updater, Dispatcher
import pickle


tfidf = pickle.load(open('tfidf', 'rb'))
clf = pickle.load(open('clf_task1', 'rb'))

import string
import re

classes = open('classes.txt', 'r', encoding="utf8").readlines()
desc = {int(class_.split(' ')[0]): class_.split('-')[1].split('/')[0] for class_ in classes}

def normalize(s):
    s = s.lower()
    return re.sub(r"[\W_]+", " ", s)

updater = Updater(token='-', use_context=True)
dispatcher = updater.dispatcher

def echo(update, context):
    X_test = tfidf.transform([normalize(update.message.text)])
    pred = clf.predict(X_test)

    res = desc[pred[0]]

    context.bot.send_message(chat_id=update.effective_chat.id, text='Category is :'+res+', category id:' + str(pred[0]))


from telegram.ext import MessageHandler, Filters
echo_handler = MessageHandler(Filters.text, echo)
dispatcher.add_handler(echo_handler)
updater.start_polling()