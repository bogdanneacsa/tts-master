# -*- coding: utf-8 -*-

import codecs
file_in = codecs.open('DATA.csv', 'r', encoding='utf-8')
file_out = open('ise_processed', 'w')

sentence_opened = False
sentence = ''
sentiment = ''
sent_id = 0
available_sentiments = []

for row in file_in:
    if not sentence_opened and row.find(',') > 0:
        sentence_opened = True
        sentiment = row[0:row.find(',')]
        row = row[(row.find(',')+1):]
        sent_id += 1
    if not row.endswith(u'á\n'):
        sentence += row
        file_out.write('%i---%s---%s'%(sent_id, sentiment, sentence))
        sentence_opened = False
        sentence = ''
        available_sentiments.append(sentiment)
    else:
        sentence += row.replace(u'á\n', '')
        
file_in.close()
file_out.close()
print set(available_sentiments)