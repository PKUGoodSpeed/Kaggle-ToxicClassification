from googletrans import Translator
import re
import time
import random

f = open('kaggle materials/new_test.csv', encoding='utf-8', errors='ignore')
#f = open('train_sample.txt', encoding='utf-8', errors='ignore')
fo = open('kaggle materials/new_test_transed.csv', 'a')
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          "]+", flags=re.UNICODE)

translator = Translator()
#fo.write(f.readline())
f.readline()
line_cnt = 1
for line in f.readlines():
	line_cnt += 1
	
	if (line_cnt < 83541):
		continue

	line_ = line.split(',')
	text = emoji_pattern.sub(r'', line_[1])
	text_transed = ''
	for attempts in range( 10):
		try:
			text_transed = translator.translate(text).text
			break
		except:
			print('error %i' % line_cnt)

	if (attempts == 9):
		sys.exit("some error") 

	if (line_cnt % 5000 == 0):
		print (line_cnt)
	
	line_[1] = text_transed
	new_line = ','.join(line_)
	if (not new_line.endswith('\n')):
		new_line += '\n'
	fo.write(new_line)
	time.sleep(1*random.random())
	#print (dd.text)
