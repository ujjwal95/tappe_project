from glob import glob
import pandas as pd
from nltk.stem import PorterStemmer

df = pd.read_csv('/Users/karanjani/Desktop/weightLoss_Cleaned.csv')

docIndex = df.index[(df['speakerID'] == 'doctor')].tolist()
ptIndex = df.index[(df['speakerID'] == 'patient')].tolist()

ptIndex = [x-1 for x in ptIndex]
temp = sorted(docIndex + ptIndex)

suspicion = sorted(list(set([x for x in temp if temp.count(x)>1])))


stringList = []
metricList = []
wordCount = []
semanticList = []
ptResponse1 = []
ptResponse2 = []

# UNSURE IF LENGTH OF PATIENT RESPONSE SHOULD BE FACTORED IN; SOME SPs are WAY MORE LONG-WINDED THAN OTHERS; COULD BE POINT OF VALIDATION LATER?

ps = PorterStemmer()

stopwords = ['um','a','the','uh','an','i','okay','yeah','alright']
keywords = ['what','tell','describe','how','why','who','could']
penaltyWords = ['where','when']
penaltyTuples = [('how','long'),('how','much'),('how','often'),('how','tall'),('how','heavy'),('how','many'),('how','frequent')]
for index in suspicion:
	docString = df.get_value(index,'stringList').replace('\'s','').replace('\'d','').replace('\'re','').replace('\'ll','') #Clean contractions
	docCleaned = [ps.stem(token) for token in docString.lower().split(' ') if token not in stopwords]
	docBigram = zip(docCleaned, docCleaned[1:])
	points = float(len(set(keywords)&set(docCleaned)))
	penalty = float(len(set(penaltyWords)&set(docCleaned))) + float(len(set(penaltyTuples)&set(docBigram)))

	score = points - penalty
	# if len(df.get_value(index+1,'stringList').lower().split(' ')) == 1: #RATHER THAN BOOST LONG RESPONSES, THIS PENALIZES SHORT RESPONSES (beneficial for sensitivity/specificity)
	# 	overlap = 0
	# if len(df.get_value(index+1,'stringList').lower().split(' ')) < 4:
	# 	overlap = overlap - 0.3

	answerlen = len(df.get_value(index+1,'stringList').lower().split(' '))

	try:
		if df.loc[index+2,'speakerID'] == 'patient':
			answerlen2 = len(df.get_value(index+2,'stringList').lower().split(' '))
		else:
			answerlen2 = 0
	except KeyError:
		print "Key error encountered"


	if len(docCleaned) > 0:
		metric = round(float(score)/float(len(docCleaned)),4) #HOW DOES THE LENGTH OF THE SENTENCE AFFECT THE METRIC. IS THIS FAIR??? 
		# MAYBE IT'S THE RATIO OF THE DOCTOR QUESTION TO PATIENT RESPONSE THAT MATTERS FOR THIS METRIC? DOES PATIENT RESPONSE INCLUDE JUST THE NEXT SPEAKER TURN OR SHOULD THERE BE MORE INCLUDED?
		# FOR EXAMPLE PATIENT COULD START WITH 'I DON'T KNOW' AND THIS WRECKS YOUR RATIO.
		# WHEN YOU GET TIME DATA, CAN YOU USE LENGTH TIL RESPONSE AS A METRIC FOR OPEN OR CONCEPTUALLY OPEN? 

		count = len(docCleaned)
		#SHORT RESPONSE HAVE NEGATIVE VALUES; RATHER THAN FINDING OPEN QUESTIONS, WHAT IF WE ARE ABLE TO CREATE 3 CATEGORIES: CLOSED, QUESTIONABLY OPEN, OPEN
		stringList.append(df.get_value(index,'stringList').lower())
		semanticList.append(df.get_value(index,'semanticType'))
		ptResponse1 += [answerlen]
		ptResponse2 += [answerlen2]
		metricList += [score]
		wordCount += [count]
		# print df.get_value(index,'stringList').lower() + ' | ', metric

df1 = pd.DataFrame({'score':metricList,'stringList':stringList,'semanticType':semanticList,'ptResponse1':ptResponse1,'ptResponse2':ptResponse2,'wordCount':wordCount})
df1 = df1.reindex_axis(['stringList','score','wordCount','ptResponse1','ptResponse2','semanticType'], axis=1)
df1.to_csv('openQmetric_weightLoss.csv', index=False)


