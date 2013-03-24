import os
from gensim import corpora, models, similarities

from xml.dom.minidom import parseString

class Sentence():
    
    def __init__(self, sentence, valence):
        self.sentence = sentence
        self.valence = valence


class SemevalCorpus():
    
    def __init__(self, data_file, valence_file):
        self.sentence_data = self.parse_semeval_data(data_file, valence_file)
        self.storage_file = 'corpus_%s.mm'%(id(self),)
        # TODO: use actual stemmers and stowords removal from either NLTK or stemming module
        dictionary = corpora.Dictionary(entry.sentence.lower().split() for entry in self.sentence_data.values())
        stoplist = set('for a of the and to in'.split())
        stop_ids = [dictionary.token2id[stopword] for stopword in stoplist
                                                            if stopword in dictionary.token2id]
        once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
        dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
        dictionary.compactify()
        self.dictionary = dictionary
        
    def __del__(self):
        if os.path.exists(self.storage_file):
            self.clear()
        
    def parse_semeval_data(self, xml_file, valence_file):
        results_dict = {}
        data_string = open(xml_file, 'r').read()
        dom = parseString(data_string)
        instances = dom.getElementsByTagName('instance')
        for instance in instances:
            results_dict[instance.attributes['id'].nodeValue] = {'value' : instance.firstChild.nodeValue}
        for line in open(valence_file):
            sent_id, valence = line.split()
            results_dict[sent_id]['valence'] = valence
        for key in results_dict:
            results_dict[key] = Sentence(results_dict[key]['value'], results_dict[key]['valence'])
        return results_dict
        
    def sentence_for_id(self, sent_id):
        return self.sentence_data.get(sent_id, None).sentence
    
    def vector_for_id(self, sent_id):
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
            
    def to_sparse_arff(self, output_file, corpus_type='tfidf'):
        if corpus_type == 'tfidf':
            tfidf = models.TfidfModel(self.mm_corpus())
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
                fp.write("@attribute valence numeric\n\n")
                fp.write("@data\n")
                for sent_id in self.sentence_data:
                    sparse_vector_rep = tfidf[self.vector_for_id(sent_id)]
                    data_string = '{'
                    for entry in sparse_vector_rep:
                        data_string += '%s %s,'%(entry[0], entry[1])
                    data_string += '%s %s}\n'%(len(words), self.sentence_data[sent_id].valence)
                    fp.write(data_string)
                
                    
            
    def mm_corpus(self):
        if not os.path.exists(self.storage_file):
            self.serialize()
        return corpora.MmCorpus(self.storage_file)

#documents = ["Human machine interface for lab abc computer applications",
#             "A survey of user opinion of computer system response time",
#             "The EPS user interface management system",
#             "System and human system engineering testing of EPS",
#             "Relation of user perceived response time to error measurement",
#             "The generation of random binary unordered trees",
#             "The intersection graph of paths in trees",
#             "Graph minors IV Widths of trees and well quasi ordering",
#             "Graph minors A survey"]
#
#stoplist = set('for a of the and to in'.split())
#texts = [[word for word in document.lower().split() if word not in stoplist]
#            for document in documents]
#
## remove words that appear only once
#all_tokens = sum(texts, [])
#tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
#texts = [[word for word in text if word not in tokens_once]
#          for text in texts]
#
#dictionary = corpora.Dictionary(texts)
#new_doc = "Human computer interaction"
#new_vec = dictionary.doc2bow(new_doc.lower().split())

corpus_memory_friendly = SemevalCorpus('affectivetext_trial.xml', 'affectivetext_trial.valence.gold')
corpus_memory_friendly.to_sparse_arff('sample.arff')

#mm_corpus = corpus_memory_friendly.mm_corpus()
#tfidf = models.TfidfModel(mm_corpus)
#
##print corpus_memory_friendly.sentence_for_id('1')
#print corpus_memory_friendly.vector_for_id('1')
#print tfidf[corpus_memory_friendly.vector_for_id('1')]
#    
#corpora.MmCorpus.serialize('corpus.mm', corpus_memory_friendly)
#corpus = corpora.MmCorpus('corpus.mm')
#print corpus
#
#tfidf = models.TfidfModel(corpus)
##input_txt = corpus.doc2bow("Once upon a time there was a village shop.".lower().split())
#for doc in tfidf[corpus]:
#    print doc

#print new_vec
