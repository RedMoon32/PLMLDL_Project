import pandas as pd
import pickle


test = pd.read_parquet('data/task1_test_for_user.parquet')

print('unique_code_1323')

tfidf = pickle.load(open('tfidf', 'rb'))
clf = pickle.load(open('clf_task1', 'rb'))

import string
import re

def normalize(s):
    s = s.lower()
    return re.sub(r"[\W_]+", " ", s)

test['item_name'] = test['item_name'].apply(normalize)


X_test = tfidf.transform(test.item_name)

pred = clf.predict(X_test)

res = pd.DataFrame(pred, columns=['pred'])
res['id'] = test['id']

res[['id', 'pred']].to_csv('answers.csv', index=None)
