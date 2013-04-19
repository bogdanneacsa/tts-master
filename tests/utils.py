import os
from gensim import corpora, models, similarities
import nltk
import nltk.corpus as sw_corpus
import nltk.stem.porter as porter

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
        
        
class FairytaleCorpus():
    
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
        stop_ids = [dictionary.token2id[stopword] for stopword in stopwords
                                                            if stopword in dictionary.token2id]
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
            
    def __del__(self):
        if os.path.exists(self.storage_file):
            self.clear()
        
    def preprocess_word(self, word):
        """
        Do preprocessing on a word, remove all non letter characters and stem the word.
        """
        new_word = ''.join([ch for ch in word if ch.isalnum() and not ch.isdigit()])
        return STEMMER.stem(new_word)
    
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
        return self.dictionary.doc2bow(sent.sentence.lower().split())
    
    def clear(self):
        if os.path.exists(self.storage_file):
            os.remove(self.storage_file)
        if os.path.exists(self.storage_file + '.index'):
            os.remove(self.storage_file + '.index')
    
    def serialize(self):
        corpora.MmCorpus.serialize(self.storage_file, self)
        
    def __iter__(self):
        for line in self.sentence_data.values():
            yield self.dictionary.doc2bow(line.sentence.lower().split())
            
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

grimm_data_folders = [os.sep.join(['..', 'fairytales', 'Grimms', 'emmood']), os.sep.join(['..', 'fairytales', 'HCAndersen', 'emmood']), os.sep.join(['..', 'fairytales', 'Potter', 'emmood']), ]
fairy_corpus = FairytaleCorpus(grimm_data_folders)
#fairy_corpus.to_sparse_arff('fairytale_balance_tfidf.arff', 'tfidf')
fairy_corpus.to_binary_sparse_arff('binary_balance_fairytale.arff')
#fairy_corpus.to_sparse_arff('fairytale_lsi.arff', 'lsi')
#fairy_corpus.to_sparse_arff('fairytale_lda.arff', 'lda')


