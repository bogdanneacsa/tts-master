import os
from gensim import corpora, models, similarities
import nltk
import nltk.corpus as sw_corpus
import nltk.stem.porter as porter
from nltk.stem.wordnet import WordNetLemmatizer
from senti_classifier import senti_classifier

senti_anal = senti_classifier.polarity_scores

LMTZR = WordNetLemmatizer()

DO_STEMMING = False
DO_STOPWORDS = False
STEMMER = porter.PorterStemmer()

POS = ("H", "Su+")
NEG = ("A", "D", "F", "Sa", "Su-")

class Sentence():
    
    def __init__(self, sentence, valence=None, primary_emotion=None, mood=None):
        self.sentence = sentence
        self.valence = valence
        self.primary_emotion = primary_emotion
        self.mood = mood
        tokens = nltk.word_tokenize(sentence)
        self.tagged_sent = nltk.pos_tag(tokens)
        
        
    def get_turney_feat1(self):
        """
        Look for the following two-word phrases:
        JJ followed by NN or NNS followed by anything
        """
        for idx, val in enumerate(self.tagged_sent[:-1]):
            if val[1] == 'JJ' and self.tagged_sent[idx+1][1] in ('NN', 'NNS'):
                return 1
        return 0
    
    
    def get_turney_feat2(self):
        """
        Look for the following two-word phrases:
        RB, RBR or RBS followed by JJ followed by not NN nor NNS
        """
        for idx, val in enumerate(self.tagged_sent[:-1]):
            if val[1] in ('RB', 'RBR', 'RBS') and self.tagged_sent[idx+1][1] == 'JJ':
                if idx + 2 >= len(self.tagged_sent) or self.tagged_sent[idx+2][0] not in ('NN', 'NNS'):
                    return 1
        return 0
    
    
    def get_turney_feat3(self):
        """
        Look for the following two-word phrases:
        JJ followed by JJ followed by not NN nor NNS
        """
        for idx, val in enumerate(self.tagged_sent[:-1]):
            if val[1] == 'JJ' and self.tagged_sent[idx+1][1] == 'JJ':
                if idx + 2 >= len(self.tagged_sent) or self.tagged_sent[idx+2][0] not in ('NN', 'NNS'):
                    return 1
        return 0
    
    
    def get_turney_feat4(self):
        """
        Look for the following two-word phrases:
        NN or NNS followed by JJ followed by not NN nor NNS
        """
        for idx, val in enumerate(self.tagged_sent[:-1]):
            if val[1] in ('NN', 'NNS') and self.tagged_sent[idx+1][1] == 'JJ':
                if idx + 2 >= len(self.tagged_sent) or self.tagged_sent[idx+2][0] not in ('NN', 'NNS'):
                    return 1
        return 0
    
    
    def get_turney_feat5(self):
        """
        Look for the following two-word phrases:
        RB, RBR, RBS followed by VBN or VBG followed by not anything
        """
        for idx, val in enumerate(self.tagged_sent[:-1]):
            if val[1] in ('RB', 'RBR', 'RBS') and self.tagged_sent[idx+1][1] in ('VBN', 'VBG'):
                return 1
        return 0
    
        
class BaseCorpus():
    
    def __del__(self):
        if os.path.exists(self.storage_file):
            self.clear()
        
    def preprocess_word(self, word):
        """
        Do preprocessing on a word, remove all non letter characters and stem the word.
        """
        new_word = ''.join([ch for ch in word if ch.isalnum() and not ch.isdigit()])
        if DO_STEMMING:
            return STEMMER.stem(new_word)
        else:
            return LMTZR.lemmatize(new_word)
    
    def sentence_for_id(self, sent_id):
        """
        Return the sentence for this id.
        """
        return self.sentence_data.get(sent_id, None).sentence
    
    def vector_for_id(self, sent_id):
        """
        For the given sentence return the vector representation using our dictionary.
        """
        sent = self.sentence_data.get(sent_id, None)
        if sent is None:
            # TODO: Add entry here with unknown valence ??
            raise Exception("No sentence stored so far!")
        processed_sent = [self.preprocess_word(word) for word in sent.sentence.lower().split()]
        return self.dictionary.doc2bow(processed_sent)
    
    def clear(self):
        if os.path.exists(self.storage_file):
            os.remove(self.storage_file)
        if os.path.exists(self.storage_file + '.index'):
            os.remove(self.storage_file + '.index')
    
    def serialize(self):
        corpora.MmCorpus.serialize(self.storage_file, self)
        
    def __iter__(self):
        for line in self.sentence_data.values():
            processed_sent = [self.preprocess_word(word) for word in line.sentence.lower().split()]
            yield self.dictionary.doc2bow(processed_sent)
            
    def tfidf_model(self):
        return models.TfidfModel(self.mm_corpus())
            
    def lsi_model(self):
        return models.LsiModel(self.tfidf_model()[self.mm_corpus()], id2word=self.dictionary, num_topics=200)
            
    def lda_model(self):
        return models.LdaModel(self.mm_corpus(), id2word=self.dictionary, num_topics=150)
            
    def to_sparse_arff(self, output_file, corpus_type='tfidf'):
        """
        Write out the corpus in sparse arff format, with the corpus type being either:
         - tfidf, lsi, lda
        """
        max_neutr = self.nr_pos + self.nr_neg
        neutr_cnt = 0
        if corpus_type == 'tfidf':
            model = self.tfidf_model()
        elif corpus_type == 'lsi':
            model = self.lsi_model()
        elif corpus_type == 'lda':
            model = self.lda_model()
        else:
            raise Exception("Unknown type %s"%(corpus_type,))
        words = []
        for word in self.dictionary.token2id:
            words.append((word, self.dictionary.token2id[word]))
        # Sort the words after the index so we can write our arff file.
        words = sorted(words, key=lambda x: x[1])
        print "Corpus has %i words"%(len(words))
        with open(output_file, 'w') as fp:
            fp.write('@relation tfidf_features\n\n')
            for word in words:
                fp.write("@attribute %s numeric\n"%(word[0]))
            fp.write("@attribute turney_rel_1 {0, 1}\n")
            fp.write("@attribute turney_rel_2 {0, 1}\n")
            fp.write("@attribute turney_rel_3 {0, 1}\n")
            fp.write("@attribute turney_rel_4 {0, 1}\n")
            fp.write("@attribute turney_rel_5 {0, 1}\n")
            fp.write("@attribute primary_emotion {Neg, N, Pos}\n\n")
            fp.write("@data\n")
            for sent_id in self.sentence_data:
                sparse_vector_rep = model[self.vector_for_id(sent_id)]
                emotion = self.sentence_data[sent_id].primary_emotion
                if sparse_vector_rep and (emotion != 'N' or neutr_cnt < max_neutr):
                    data_string = '{'
                    for entry in sparse_vector_rep:
                        data_string += '%s %s,'%(entry[0], entry[1])
                    # Add turney specific attributes
                    if self.sentence_data[sent_id].get_turney_feat1():
                        data_string += '%s %s,'%(len(words), 1)
                    if self.sentence_data[sent_id].get_turney_feat2():
                        data_string += '%s %s, '%(len(words) + 1, 1)
                    if self.sentence_data[sent_id].get_turney_feat3():
                        data_string += '%s %s, '%(len(words) + 2, 1)
                    if self.sentence_data[sent_id].get_turney_feat4():
                        data_string += '%s %s, '%(len(words) + 3, 1)
                    if self.sentence_data[sent_id].get_turney_feat5():
                        data_string += '%s %s, '%(len(words) + 4, 1)
                    data_string += '%s %s}\n'%(len(words) + 5, self.sentence_data[sent_id].primary_emotion)
                    fp.write(data_string)
                    if emotion == 'N':
                        neutr_cnt += 1
                    
                    
    def to_binary_sparse_arff(self, output_file):
        max_neutr = self.nr_pos + self.nr_neg
        neutr_cnt = 0
        words = []
        for word in self.dictionary.token2id:
            words.append((word, self.dictionary.token2id[word]))
        with open(output_file, 'w') as fp:
            fp.write('@relation binary_features\n\n')
            for word in words:
                fp.write("@attribute %s {0, 1}\n"%(word[0]))
            fp.write("@attribute turney_rel_1 {0, 1}\n")
            fp.write("@attribute turney_rel_2 {0, 1}\n")
            fp.write("@attribute turney_rel_3 {0, 1}\n")
            fp.write("@attribute turney_rel_4 {0, 1}\n")
            fp.write("@attribute turney_rel_5 {0, 1}\n")
            fp.write("@attribute primary_emotion {Neg,N,Pos}\n\n")
            fp.write("@data\n")
            for sent_id in self.sentence_data:
                binary_vec = self.vector_for_id(sent_id)
                emotion = self.sentence_data[sent_id].primary_emotion
                if binary_vec and (emotion != 'N' or neutr_cnt < max_neutr):
                    data_string = '{'
                    for entry in binary_vec:
                        data_string += '%s 1,'%(entry[0],)
                    # Add turney specific attributes
                    if self.sentence_data[sent_id].get_turney_feat1():
                        data_string += '%s %s,'%(len(words), 1)
                    if self.sentence_data[sent_id].get_turney_feat2():
                        data_string += '%s %s, '%(len(words) + 1, 1)
                    if self.sentence_data[sent_id].get_turney_feat3():
                        data_string += '%s %s, '%(len(words) + 2, 1)
                    if self.sentence_data[sent_id].get_turney_feat4():
                        data_string += '%s %s, '%(len(words) + 3, 1)
                    if self.sentence_data[sent_id].get_turney_feat5():
                        data_string += '%s %s, '%(len(words) + 4, 1)
                    data_string += '%s %s}\n'%(len(words) + 5, self.sentence_data[sent_id].primary_emotion)
                    fp.write(data_string)
                    if emotion == 'N':
                        neutr_cnt += 1
                    
            
    def mm_corpus(self):
        if not os.path.exists(self.storage_file):
            self.serialize()
        return corpora.MmCorpus(self.storage_file)
    
    
class FairytaleCorpus(BaseCorpus):
    
    def __init__(self, data_folders):
        dict_data = {}
        start_idx = 0
        self.nr_pos = self.nr_neg = self.nr_neu = 0
        for data_folder in data_folders:
            for data_file in os.listdir(data_folder):
                results_dict, end_idx = self.parse_fairytale_data(os.path.join(data_folder, data_file), start_idx)
                dict_data.update(results_dict)
                start_idx += end_idx
        self.sentence_data = dict_data
        self.storage_file = 'corpus_%s.mm'%(id(self),)
        data = [[self.preprocess_word(word) for word in entry.sentence.lower().split()] for entry in self.sentence_data.values()]
        dictionary = corpora.Dictionary(data)
        stopwords = sw_corpus.stopwords.words('english')
        #Just remove empty string here
        stopwords.append('')
        if DO_STOPWORDS:
            stop_ids = [dictionary.token2id[stopword] for stopword in stopwords
                                                            if stopword in dictionary.token2id]
        else:
            stop_ids = [dictionary.token2id['stopword']]
        once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
        dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
        dictionary.compactify()
        self.dictionary = dictionary
    
    
    def parse_fairytale_data(self, file_name, start_idx=0):
        """
        For this file name, parse all sentences and return a dictionary
        of the form {sentence_id : Sentence()}
        """
        results_dict = {}
        for line in open(file_name):
            sent_id, sentence = self.process_line(line)
            results_dict[str(int(sent_id)+start_idx)] = sentence
        return results_dict, int(sent_id) + start_idx
    
    
    def process_line(self, line):
        """
        Process a given line from the fairytale corpus, return a tuple in the form
        (sent_id : Sentence() entity)
        """
        first_index = line.index(':')
        sent_id = line[:first_index]
        second_index = first_index + line[first_index+1:].index(':') + 1
        primary_emo = line[first_index:second_index].split('\t')[-1]
        third_index = second_index + line[second_index+1:].index(':') + 1
        mood = line[second_index:third_index].split('\\t')[-1]
        sentence = line[third_index+1:].split('\t')[1]
        if primary_emo in NEG:
            primary_emo = 'Neg'
            self.nr_neg += 1
        elif primary_emo in POS:
            primary_emo = 'Pos'
            self.nr_pos += 1
        else:
            self.nr_neu += 1
        return sent_id, Sentence(sentence, primary_emotion=primary_emo, mood=mood)
    
    
class ISEARCorpus(BaseCorpus):
    #[u'joy', u'shame', u'sadness', u'guilt', u'disgust', u'anger', u'fear']
    def __init__(self, data_folders, prefix='ise_'):
        dict_data = {}
        start_idx = 0
        for data_folder in data_folders:
            for data_file in os.listdir(data_folder):
                if data_file.startswith(prefix):
                    results_dict, end_idx = self.parse_fairytale_data(os.path.join(data_folder, data_file), start_idx)
                    dict_data.update(results_dict)
                    start_idx += end_idx
        self.sentence_data = dict_data
        self.storage_file = 'corpus_%s.mm'%(id(self),)
        data = [[self.preprocess_word(word) for word in entry.sentence.lower().split()] for entry in self.sentence_data.values()]
        dictionary = corpora.Dictionary(data)
        stopwords = sw_corpus.stopwords.words('english')
        #Just remove empty string here
        stopwords.append('')
        if DO_STOPWORDS:
            stop_ids = [dictionary.token2id[stopword] for stopword in stopwords
                                                            if stopword in dictionary.token2id]
        else:
            stop_ids = [dictionary.token2id['']]
        once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
        dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
        dictionary.compactify()
        self.dictionary = dictionary
    
    
    def parse_fairytale_data(self, file_name, start_idx=0):
        """
        For this file name, parse all sentences and return a dictionary
        of the form {sentence_id : Sentence()}
        """
        results_dict = {}
        for line in open(file_name):
            sent_id, sentence = self.process_line(line)
            results_dict[str(int(sent_id)+start_idx)] = sentence
        return results_dict, int(sent_id) + start_idx
    
    
    def process_line(self, line):
        """
        Process a given line from the fairytale corpus, return a tuple in the form
        (sent_id : Sentence() entity)
        """
        sent_id, primary_emo, sentence = line.split('---')
        return sent_id, Sentence(sentence, primary_emotion=primary_emo)
    
    
    def to_sparse_arff(self, output_file, corpus_type='tfidf'):
        """
        Write out the corpus in sparse arff format, with the corpus type being either:
         - tfidf, lsi, lda
        """
        if corpus_type == 'tfidf':
            model = self.tfidf_model()
        elif corpus_type == 'lsi':
            model = self.lsi_model()
        elif corpus_type == 'lda':
            model = self.lda_model()
        else:
            raise Exception("Unknown type %s"%(corpus_type,))
        words = []
        for word in self.dictionary.token2id:
            words.append((word, self.dictionary.token2id[word]))
        # Sort the words after the index so we can write our arff file.
        words = sorted(words, key=lambda x: x[1])
        print "Corpus has %i words"%(len(words))
        with open(output_file, 'w') as fp:
            fp.write('@relation tfidf_features\n\n')
            for word in words:
                fp.write("@attribute %s numeric\n"%(word[0]))
            fp.write("@attribute primary_emotion {joy,shame,sadness,guilt,disgust,anger,fear}\n\n")
            fp.write("@data\n")
            for sent_id in self.sentence_data:
                sparse_vector_rep = model[self.vector_for_id(sent_id)]
                if sparse_vector_rep:
                    data_string = '{'
                    for entry in sparse_vector_rep:
                        data_string += '%s %s,'%(entry[0], entry[1])
                    data_string += '%s %s}\n'%(len(words), self.sentence_data[sent_id].primary_emotion)
                    fp.write(data_string)
                    
                    
    def to_binary_sparse_arff(self, output_file):
        words = []
        for word in self.dictionary.token2id:
            words.append((word, self.dictionary.token2id[word]))
        with open(output_file, 'w') as fp:
            fp.write('@relation binary_features\n\n')
            for word in words:
                fp.write("@attribute %s {0, 1}\n"%(word[0]))
            fp.write("@attribute primary_emotion {joy,shame,sadness,guilt,disgust,anger,fear}\n\n")
            fp.write("@data\n")
            for sent_id in self.sentence_data:
                binary_vec = self.vector_for_id(sent_id)
                if binary_vec:
                    data_string = '{'
                    for entry in binary_vec:
                        data_string += '%s 1,'%(entry[0],)
                    # Add turney specific attributes
                    data_string += '%s %s}\n'%(len(words), self.sentence_data[sent_id].primary_emotion)
                    fp.write(data_string)


class TwoWordTurneyISEARCorpus(ISEARCorpus):
    
    def is_turney_feat1(self, ps1, ps2, ps3=None):
        """
        Look for the following two-word phrases:
        JJ followed by NN or NNS followed by anything
        """
        if ps1 == 'JJ' and ps2 in ('NN', 'NNS'):
            return 1
        return 0
    
    
    def is_turney_feat2(self, ps1, ps2, ps3=None):
        """
        Look for the following two-word phrases:
        RB, RBR or RBS followed by JJ followed by not NN nor NNS
        """
        if ps1 in ('RB', 'RBR', 'RBS') and ps2 == 'JJ':
            if ps3 not in ('NN', 'NNS'):
                return 1
        return 0
    
    
    def is_turney_feat3(self, ps1, ps2, ps3=None):
        """
        Look for the following two-word phrases:
        JJ followed by JJ followed by not NN nor NNS
        """
        if ps1 == 'JJ' and ps2 == 'JJ':
            if ps3 not in ('NN', 'NNS'):
                return 1
        return 0
    
    
    def is_turney_feat4(self, ps1, ps2, ps3=None):
        """
        Look for the following two-word phrases:
        NN or NNS followed by JJ followed by not NN nor NNS
        """
        if ps1 in ('NN', 'NNS') and ps2 == 'JJ':
            if ps3 not in ('NN', 'NNS'):
                return 1
        return 0
    
    
    def is_turney_feat5(self, ps1, ps2, ps3=None):
        """
        Look for the following two-word phrases:
        RB, RBR, RBS followed by VBN or VBG followed by not anything
        """
        if ps1 in ('RB', 'RBR', 'RBS') and ps2 in ('VBN', 'VBG'):
            return 1
        return 0
    
    
    def is_turney(self, ps1, ps2, ps3=None):
        return (self.is_turney_feat1(ps1, ps2, ps3) or self.is_turney_feat2(ps1, ps2, ps3)
                or self.is_turney_feat3(ps1, ps2, ps3) or self.is_turney_feat4(ps1, ps2, ps3)
                or self.is_turney_feat5(ps1, ps2, ps3))
        
        
    def vector_for_id(self, sent_id):
        """
        For the given sentence return the vector representation using our dictionary.
        """
        sent = self.sentence_data.get(sent_id, None)
        if sent is None:
            # TODO: Add entry here with unknown valence ??
            raise Exception("No sentence stored so far!")
        processed_sent = []
        split_sent = sent.sentence.lower().split()
        for idx in range(len(split_sent) - 1):
            processed_sent.append(split_sent[idx] + ' ' + split_sent[idx+1])
        return self.dictionary.doc2bow(processed_sent)
    
    
    def __iter__(self):
        for line in self.sentence_data.values():
            processed_sent = []
            split_sent = line.sentence.lower().split()
            for idx in range(len(split_sent) - 1):
                processed_sent.append(split_sent[idx] + ' ' + split_sent[idx+1])
            yield self.dictionary.doc2bow(processed_sent)
    
    
    #[u'joy', u'shame', u'sadness', u'guilt', u'disgust', u'anger', u'fear']
    def __init__(self, data_folders, prefix='ise_'):
        dict_data = {}
        start_idx = 0
        for data_folder in data_folders:
            for data_file in os.listdir(data_folder):
                if data_file.startswith(prefix):
                    results_dict, end_idx = self.parse_fairytale_data(os.path.join(data_folder, data_file), start_idx)
                    dict_data.update(results_dict)
                    start_idx += end_idx
        self.sentence_data = dict_data
        self.storage_file = 'corpus_%s.mm'%(id(self),)
        data = []
        for entry in self.sentence_data.values():
            sent_data = []
            sentence = entry.sentence.lower()
            tokens = nltk.word_tokenize(sentence)
            tagged_sent = nltk.pos_tag(tokens)
            for idx in range(len(tagged_sent) - 1):
                if ((idx < len(tagged_sent) - 2 and (self.is_turney(tagged_sent[idx][1], tagged_sent[idx+1][1], tagged_sent[idx+2][1])))
                        or 
                    (idx >= len(tagged_sent) - 2 and (self.is_turney(tagged_sent[idx][1], tagged_sent[idx+1][1])))):
                    sent_data.append(tagged_sent[idx][0] + ' ' + tagged_sent[idx+1][0])
            data.append(sent_data)
        dictionary = corpora.Dictionary(data)
        dictionary.compactify()
        self.dictionary = dictionary
    
    
    def parse_fairytale_data(self, file_name, start_idx=0):
        """
        For this file name, parse all sentences and return a dictionary
        of the form {sentence_id : Sentence()}
        """
        results_dict = {}
        for line in open(file_name):
            sent_id, sentence = self.process_line(line)
            results_dict[str(int(sent_id)+start_idx)] = sentence
        return results_dict, int(sent_id) + start_idx
    
    
    def process_line(self, line):
        """
        Process a given line from the fairytale corpus, return a tuple in the form
        (sent_id : Sentence() entity)
        """
        sent_id, primary_emo, sentence = line.split('---')
        if DO_STOPWORDS:
            stopwords = sw_corpus.stopwords.words('english')
        else:
            stopwords = []
        #Just remove empty string here
        stopwords.append('')
        sentence = ' '.join([self.preprocess_word(word) for word in sentence.lower().split() if word not in stopwords])
        return sent_id, Sentence(sentence, primary_emotion=primary_emo)
    
    
    def to_sparse_arff(self, output_file, corpus_type='tfidf'):
        """
        Write out the corpus in sparse arff format, with the corpus type being either:
         - tfidf, lsi, lda
        """
        if corpus_type == 'tfidf':
            model = self.tfidf_model()
        elif corpus_type == 'lsi':
            model = self.lsi_model()
        elif corpus_type == 'lda':
            model = self.lda_model()
        else:
            raise Exception("Unknown type %s"%(corpus_type,))
        words = []
        for word in self.dictionary.token2id:
            words.append((word, self.dictionary.token2id[word]))
        # Sort the words after the index so we can write our arff file.
        words = sorted(words, key=lambda x: x[1])
        print "Corpus has %i words"%(len(words))
        with open(output_file, 'w') as fp:
            fp.write('@relation tfidf_features\n\n')
            for word in words:
                fp.write("@attribute %s numeric\n"%(word[0].replace(' ', '_')))
            fp.write("@attribute primary_emotion {joy,shame,sadness,guilt,disgust,anger,fear}\n\n")
            fp.write("@data\n")
            for sent_id in self.sentence_data:
                sparse_vector_rep = model[self.vector_for_id(sent_id)]
                if sparse_vector_rep:
                    data_string = '{'
                    for entry in sparse_vector_rep:
                        data_string += '%s %s,'%(entry[0], entry[1])
                    data_string += '%s %s}\n'%(len(words), self.sentence_data[sent_id].primary_emotion)
                    fp.write(data_string)
                    
                    
    def to_binary_sparse_arff(self, output_file):
        words = []
        for word in self.dictionary.token2id:
            words.append((word, self.dictionary.token2id[word]))
        with open(output_file, 'w') as fp:
            fp.write('@relation binary_features\n\n')
            for word in words:
                fp.write("@attribute %s {0, 1}\n"%(word[0].replace(' ', '_')))
            fp.write("@attribute primary_emotion {joy,shame,sadness,guilt,disgust,anger,fear}\n\n")
            fp.write("@data\n")
            for sent_id in self.sentence_data:
                binary_vec = self.vector_for_id(sent_id)
                if binary_vec:
                    data_string = '{'
                    for entry in binary_vec:
                        data_string += '%s 1,'%(entry[0],)
                    # Add turney specific attributes
                    data_string += '%s %s}\n'%(len(words), self.sentence_data[sent_id].primary_emotion)
                    fp.write(data_string)
    
    
class TwoWordISEARCorpus(TwoWordTurneyISEARCorpus):
    
    def is_turney(self, ps1, ps2, ps3=None):
        return True
    

class OnePlusTurneyISEARCorpus(TwoWordTurneyISEARCorpus, ISEARCorpus):
    
    def get_sent(self, ps1, ps2, ps3=None):
        sent = ps1[0] + ' ' + ps2[0]
        if ps3:
            sent += ' ' + ps3[0]
        sentiments = senti_anal([sent])
        if sentiments[0] > sentiments[1]:
            return sentiments[0]
        return -sentiments[1]
    
    def get_turney_feat1(self, ps1, ps2, ps3=None):
        """
        Look for the following two-word phrases:
        JJ followed by NN or NNS followed by anything
        """
        if ps1[1] == 'JJ' and ps2[1] in ('NN', 'NNS'):
            return self.get_sent(ps1, ps2, ps3)
        return None
    
    
    def get_turney_feat2(self, ps1, ps2, ps3=None):
        """
        Look for the following two-word phrases:
        RB, RBR or RBS followed by JJ followed by not NN nor NNS
        """
        if ps1[1] in ('RB', 'RBR', 'RBS') and ps2[1] == 'JJ':
            if ps3 and ps3[1] not in ('NN', 'NNS'):
                return self.get_sent(ps1, ps2, ps3)
        return None
    
    
    def get_turney_feat3(self, ps1, ps2, ps3=None):
        """
        Look for the following two-word phrases:
        JJ followed by JJ followed by not NN nor NNS
        """
        if ps1[1] == 'JJ' and ps2[1] == 'JJ':
            if ps3 and ps3[1] not in ('NN', 'NNS'):
                return self.get_sent(ps1, ps2, ps3)
        return None
    
    
    def get_turney_feat4(self, ps1, ps2, ps3=None):
        """
        Look for the following two-word phrases:
        NN or NNS followed by JJ followed by not NN nor NNS
        """
        if ps1[1] in ('NN', 'NNS') and ps2[1] == 'JJ':
            if ps3 and ps3[1] not in ('NN', 'NNS'):
                return self.get_sent(ps1, ps2, ps3)
        return None
    
    
    def get_turney_feat5(self, ps1, ps2, ps3=None):
        """
        Look for the following two-word phrases:
        RB, RBR, RBS followed by VBN or VBG followed by not anything
        """
        if ps1[1] in ('RB', 'RBR', 'RBS') and ps2[1] in ('VBN', 'VBG'):
            return self.get_sent(ps1, ps2, ps3)
        return None
    
    
    def vector_for_id(self, sent_id):
        """
        For the given sentence return the vector representation using our dictionary.
        """
        sent = self.sentence_data.get(sent_id, None)
        if sent is None:
            # TODO: Add entry here with unknown valence ??
            raise Exception("No sentence stored so far!")
        stopwords = sw_corpus.stopwords.words('english')
        #Just remove empty string here
        stopwords.append('')
        processed_sent = [word for word in sent.sentence.lower().split() if word not in stopwords]
        split_sent = sent.sentence.lower().split()
        for idx in range(len(split_sent) - 1):
            processed_sent.append(split_sent[idx] + ' ' + split_sent[idx+1])
        return self.dictionary.doc2bow(processed_sent)
    
    
    def __iter__(self):
        for line in self.sentence_data.values():
            stopwords = sw_corpus.stopwords.words('english')
            #Just remove empty string here
            stopwords.append('')
            processed_sent = [word for word in line.sentence.lower().split() if word not in stopwords]
            split_sent = line.sentence.lower().split()
            for idx in range(len(split_sent) - 1):
                processed_sent.append(split_sent[idx] + ' ' + split_sent[idx+1])
            yield self.dictionary.doc2bow(processed_sent)
            
            
    #[u'joy', u'shame', u'sadness', u'guilt', u'disgust', u'anger', u'fear']
    def __init__(self, data_folders, prefix='ise_'):
        dict_data = {}
        start_idx = 0
        for data_folder in data_folders:
            for data_file in os.listdir(data_folder):
                if data_file.startswith(prefix):
                    results_dict, end_idx = self.parse_fairytale_data(os.path.join(data_folder, data_file), start_idx)
                    dict_data.update(results_dict)
                    start_idx += end_idx
        self.sentence_data = dict_data
        self.storage_file = 'corpus_%s.mm'%(id(self),)
        data = []
        stopwords = sw_corpus.stopwords.words('english')
        #Just remove empty string here
        stopwords.append('')
        for entry in self.sentence_data.values():
            sent_data = [word for word in nltk.word_tokenize(entry.sentence) if word not in stopwords]
            data.append(sent_data)
        dictionary = corpora.Dictionary(data)
        dictionary.compactify()
        self.dictionary = dictionary
        
        
    def to_sparse_arff(self, output_file, corpus_type='tfidf'):
        """
        Write out the corpus in sparse arff format, with the corpus type being either:
         - tfidf, lsi, lda
        """
        if corpus_type == 'tfidf':
            model = self.tfidf_model()
        elif corpus_type == 'lsi':
            model = self.lsi_model()
        elif corpus_type == 'lda':
            model = self.lda_model()
        else:
            raise Exception("Unknown type %s"%(corpus_type,))
        words = []
        for word in self.dictionary.token2id:
            words.append((word, self.dictionary.token2id[word]))
        # Sort the words after the index so we can write our arff file.
        words = sorted(words, key=lambda x: x[1])
        print "Corpus has %i words"%(len(words))
        turney_attrs = ['turney1pos', 'turney1neg', 'turney2pos', 'turney2neg', 'turney3pos','turney3neg', 'turney4pos', 'turney4neg', 'turney5pos', 'turney5neg',]
        with open(output_file, 'w') as fp:
            fp.write('@relation tfidf_features\n\n')
            for word in words:
                fp.write("@attribute %s numeric\n"%(word[0].replace(' ', '_')))
            for attr in turney_attrs:
                fp.write("@attribute %s numeric\n" % attr)
            fp.write("@attribute primary_emotion {joy,shame,sadness,guilt,disgust,anger,fear}\n\n")
            fp.write("@data\n")
            for sent_id in self.sentence_data:
                sparse_vector_rep = model[self.vector_for_id(sent_id)]
                if sparse_vector_rep:
                    data_string = '{'
                    for entry in sparse_vector_rep:
                        data_string += '%s %s,'%(entry[0], entry[1])
                        
                        
                    sentence = self.sentence_data.get(sent_id, None)
                    tagged_sent = sentence.tagged_sent
                    turney_info = {}
                    for attr in turney_attrs:
                        turney_info[attr] = None
                    for idx in range(len(tagged_sent) - 1):
                        if self.is_turney(tagged_sent[idx][1], tagged_sent[idx+1][1], tagged_sent[idx+2][1] if idx < len(tagged_sent) - 2 else None):
                            print (tagged_sent[idx][0], tagged_sent[idx+1][0], tagged_sent[idx+2][0] if idx < len(tagged_sent) - 2 else None)
                        
                        turney1 = self.get_turney_feat1(tagged_sent[idx], tagged_sent[idx+1], tagged_sent[idx+2] if idx < len(tagged_sent) - 2 else None)
                        if turney1 is not None and turney1 > 0:
                            turney_info[turney_attrs[0]] = True
#                            data_string += '%s %s,'%(len(words), 1)
                        if turney1 is not None and turney1 < 0:
                            turney_info[turney_attrs[1]] = True
#                            data_string += '%s %s,'%(len(words) + 1, 1)
                        turney2 = self.get_turney_feat2(tagged_sent[idx], tagged_sent[idx+1], tagged_sent[idx+2] if idx < len(tagged_sent) - 2 else None)
                        if turney2 is not None and turney2 > 0:
                            turney_info[turney_attrs[2]] = True
#                            data_string += '%s %s,'%(len(words) + 2, 1)
                        if turney2 is not None and turney2 < 0:
                            turney_info[turney_attrs[3]] = True
#                            data_string += '%s %s,'%(len(words) + 3, 1)
                        turney3 = self.get_turney_feat3(tagged_sent[idx], tagged_sent[idx+1], tagged_sent[idx+2] if idx < len(tagged_sent) - 2 else None)
                        if turney3 is not None and turney3 > 0:
                            turney_info[turney_attrs[4]] = True
#                            data_string += '%s %s,'%(len(words) + 4, 1)
                        if turney3 is not None and turney3 < 0:
                            turney_info[turney_attrs[5]] = True
#                            data_string += '%s %s,'%(len(words) + 5, 1)
                        turney4 = self.get_turney_feat4(tagged_sent[idx], tagged_sent[idx+1], tagged_sent[idx+2] if idx < len(tagged_sent) - 2 else None)
                        if turney4 is not None and turney4 > 0:
                            turney_info[turney_attrs[6]] = True
#                            data_string += '%s %s,'%(len(words) + 6, 1)
                        if turney4 is not None and turney4 < 0:
                            turney_info[turney_attrs[7]] = True
#                            data_string += '%s %s,'%(len(words) + 7, 1)
                        turney5 = self.get_turney_feat5(tagged_sent[idx], tagged_sent[idx+1], tagged_sent[idx+2] if idx < len(tagged_sent) - 2 else None)
                        if turney5 is not None and turney5 > 0:
                            turney_info[turney_attrs[8]] = True
#                            data_string += '%s %s,'%(len(words) + 8, 1)
                        if turney5 is not None and turney5 < 0:
                            turney_info[turney_attrs[9]] = True
#                            data_string += '%s %s,'%(len(words) + 9, 1)
                    for idx, attr in enumerate(turney_attrs):
                        if turney_info[attr]:
                            data_string += '%s %s,'%(len(words) + idx, 1)
                    data_string += '%s %s}\n'%(len(words) + 10, self.sentence_data[sent_id].primary_emotion)
                    fp.write(data_string)
        

class OnePlusTwoISEARCorpus(OnePlusTurneyISEARCorpus):
    
    def is_turney(self, ps1, ps2, ps3=None):
        return True
    
    
class MoviesReviewCorpus(BaseCorpus):
    
    def __init__(self, data_folder):
        dict_data = {}
        start_idx = 0
        self.nr_pos = self.nr_neg = self.nr_neu = 0
        data_folders = [os.path.join(data_folder, 'neg'), os.path.join(data_folder, 'pos')]
        for df in data_folders:
            results_dict, end_idx = self.parse_fairytale_data(df, start_idx, 'pos' if df.endswith('pos') else 'neg')
            dict_data.update(results_dict)
            start_idx += end_idx
        self.sentence_data = dict_data
        self.storage_file = 'corpus_%s.mm'%(id(self),)
        data = [[self.preprocess_word(word) for word in entry.sentence.lower().split()] for entry in self.sentence_data.values()]
        dictionary = corpora.Dictionary(data)
        stopwords = sw_corpus.stopwords.words('english')
        #Just remove empty string here
        stopwords.append('')
        if DO_STOPWORDS:
            stop_ids = [dictionary.token2id[stopword] for stopword in stopwords
                                                            if stopword in dictionary.token2id]
        else:
            stop_ids = [dictionary.token2id['']]
        once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
        dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
        dictionary.compactify()
        self.dictionary = dictionary
    
    
    def parse_fairytale_data(self, folder_name, start_idx=0, sentiment='neg'):
        """
        For this file name, parse all sentences and return a dictionary
        of the form {sentence_id : Sentence()}
        """
        results_dict = {}
        sent_id = 0
        for file_n in os.listdir(folder_name):
            full_path = os.path.join(folder_name, file_n)
            sentence = self.process_line(full_path, sentiment)
            results_dict[str(int(sent_id)+start_idx)] = sentence
            sent_id += 1
        return results_dict, int(sent_id) + start_idx
    
    
    def process_line(self, file_n, sentiment):
        """
        Process a given line from the fairytale corpus, return a tuple in the form
        (sent_id : Sentence() entity)
        """
        sentence = open(file_n).read().replace('\n', ' ').replace('\t', ' ')
        return Sentence(sentence, primary_emotion=sentiment, mood=sentiment)
    
    
    def to_sparse_arff(self, output_file, corpus_type='tfidf'):
        """
        Write out the corpus in sparse arff format, with the corpus type being either:
         - tfidf, lsi, lda
        """
        if corpus_type == 'tfidf':
            model = self.tfidf_model()
        elif corpus_type == 'lsi':
            model = self.lsi_model()
        elif corpus_type == 'lda':
            model = self.lda_model()
        else:
            raise Exception("Unknown type %s"%(corpus_type,))
        words = []
        for word in self.dictionary.token2id:
            words.append((word, self.dictionary.token2id[word]))
        # Sort the words after the index so we can write our arff file.
        words = sorted(words, key=lambda x: x[1])
        print "Corpus has %i words"%(len(words))
        with open(output_file, 'w') as fp:
            fp.write('@relation tfidf_features\n\n')
            for word in words:
                fp.write("@attribute %s numeric\n"%(word[0]))
            fp.write("@attribute primary_emotion {neg, pos}\n\n")
            fp.write("@data\n")
            for sent_id in self.sentence_data:
                sparse_vector_rep = model[self.vector_for_id(sent_id)]
                if sparse_vector_rep:
                    data_string = '{'
                    for entry in sparse_vector_rep:
                        data_string += '%s %s,'%(entry[0], entry[1])
                    # Add turney specific attributes
                    data_string += '%s %s}\n'%(len(words), self.sentence_data[sent_id].primary_emotion)
                    fp.write(data_string)
                    
                    
    def to_binary_sparse_arff(self, output_file):
        words = []
        for word in self.dictionary.token2id:
            words.append((word, self.dictionary.token2id[word]))
        with open(output_file, 'w') as fp:
            fp.write('@relation binary_features\n\n')
            for word in words:
                fp.write("@attribute %s {0, 1}\n"%(word[0]))
            fp.write("@attribute primary_emotion {neg,pos}\n\n")
            fp.write("@data\n")
            for sent_id in self.sentence_data:
                binary_vec = self.vector_for_id(sent_id)
                if binary_vec:
                    data_string = '{'
                    for entry in binary_vec:
                        data_string += '%s 1,'%(entry[0],)
                    # Add turney specific attributes
                    data_string += '%s %s}\n'%(len(words), self.sentence_data[sent_id].primary_emotion)
                    fp.write(data_string)
    
#movies_data_folder = os.sep.join(['..', 'movieReviews', 'txt_sentoken'])
#movies_corpus = MoviesReviewCorpus(movies_data_folder)
#movies_corpus.to_sparse_arff('movies_tfidf.arff', 'tfidf')
#movies_corpus.to_binary_sparse_arff('movies_binary.arff')
#    
    
#isear_data_folder = [os.sep.join(['..', 'ISEAR'])]
#isear_corpus = OnePlusTwoISEARCorpus(isear_data_folder)
#isear_corpus.to_sparse_arff('isear_combined_tfidf.arff', 'tfidf')
#isear_corpus.to_binary_sparse_arff('isear_combined_binary.arff')

isear_data_folder = [os.sep.join(['..', 'ISEAR'])]

#isear_corpus = OnePlusTurneyISEARCorpus(isear_data_folder, prefix='ise_processed3')
#isear_corpus.to_sparse_arff('isear_one_plus_turney_tfidf3.arff', 'tfidf')
#isear_corpus.to_binary_sparse_arff('isear_one_plus_turney_binary3.arff')

isear_corpus = OnePlusTurneyISEARCorpus(isear_data_folder, prefix='ise_processed5')
isear_corpus.to_sparse_arff('isear_one_plus_turney_tfidf_new5.arff', 'tfidf')
#isear_corpus.to_binary_sparse_arff('isear_turney_binary_new5.arff')

isear_corpus = OnePlusTurneyISEARCorpus(isear_data_folder, prefix='ise_processed7')
isear_corpus.to_sparse_arff('isear_one_plus_turney_tfidf_new7.arff', 'tfidf')
#isear_corpus.to_binary_sparse_arff('isear_one_plus_turney_binary_new7.arff')

#isear_corpus = TwoWordTurneyISEARCorpus(isear_data_folder)
#isear_corpus.to_sparse_arff('isear_2w_turney_tfidf.arff', 'tfidf')
#isear_corpus.to_binary_sparse_arff('isear_2w_turney_binary.arff')
#
#isear_corpus = TwoWordISEARCorpus(isear_data_folder)
#isear_corpus.to_sparse_arff('isear_2w_tfidf.arff', 'tfidf')
#isear_corpus.to_binary_sparse_arff('isear_2w_binary.arff')

#grimm_data_folders = [os.sep.join(['..', 'fairytales', 'Grimms', 'emmood']), os.sep.join(['..', 'fairytales', 'HCAndersen', 'emmood']), os.sep.join(['..', 'fairytales', 'Potter', 'emmood']), ]
#fairy_corpus = FairytaleCorpus(grimm_data_folders)
#fairy_corpus.to_sparse_arff('fairytale_balance_tfidf.arff', 'tfidf')
#fairy_corpus.to_binary_sparse_arff('binary_balance_fairytale.arff')
#fairy_corpus.to_sparse_arff('fairytale_lsi.arff', 'lsi')
#fairy_corpus.to_sparse_arff('fairytale_lda.arff', 'lda')


