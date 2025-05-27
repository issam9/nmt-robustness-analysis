from abc import abstractmethod
import json
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
import sacrebleu
import spacy
from spacy_lefff import POSTagger
from spacy.language import Language
import random
import lemminflect
from .utils import map_tag, get_fr_lemmas
from inflecteur import inflecteur
import tqdm
import numpy as np

inflecteur = inflecteur()
inflecteur.load_dict()

random.seed(123)
'''
Implements `morph` and `search` methods for sequence to sequence tasks. Still an abstract class since
some key methods will vary depending on the target task and model.
'''


class MorpheusSeq2Seq():
    def __init__(self):
        self.lang = 'en' if self.src_lang is None else self.src_lang
        if self.lang=='en':
            self.nlp = spacy.load("en_core_web_sm")
        elif self.lang=='fr':
            self.nlp = spacy.load("fr_core_news_sm")
            Language.factory("french_pos", func=lambda nlp, name: POSTagger())
            self.nlp.add_pipe("french_pos", name='pos_lefff')
        else:
            raise ValueError("Language not supported")
    
    def get_lemmas(self, word):
        if self.lang == 'en':
            return lemminflect.getAllLemmas(word)
        else:
            return get_fr_lemmas(word, inflecteur)

    @abstractmethod
    def get_score(self, source, reference):
        pass
    
    def get_inflections(self, orig_tokenized, pos_tagged, constrain_pos):
        have_inflections = {'NOUN', 'VERB', 'ADJ'}
        universal_to_infelect = {'NOUN': 'Nom', 'VERB': 'Verbe', 'ADJ': 'Adjectif'}
        token_inflections = [] # elements of form (i, inflections) where i is the token's position in the sequence

        for i, word in enumerate(orig_tokenized):
            lemmas = self.get_lemmas(word)
            if lemmas and pos_tagged[i][1] in have_inflections:
                word_tag = pos_tagged[i][1] if self.lang=='en' else universal_to_infelect[pos_tagged[i][1]]
                if word_tag in lemmas:
                    lemma = lemmas[word_tag][0]
                    # print("lemma: ", pos_tagged[i][1], word_tag, lemma)
                else:
                    lemma = random.choice(list(lemmas.values()))[0]

                if constrain_pos:
                    if self.lang=='en':
                        inflections = (i, list(set([infl for tup in lemminflect.getAllInflections(lemma, upos=pos_tagged[i][1]).values() for infl in tup if infl != word])), lemma)
                    else:
                        df = inflecteur.dico_transformer.loc[(inflecteur.dico_transformer.lemma == lemma) & (inflecteur.dico_transformer.index!=word)].reset_index(drop=False)
                        inflections = (i, df[df['gram'] == word_tag].part.unique().tolist(), lemma)
                else:
                    if self.lang=='en':
                        inflections = (i, list(set([infl for tup in lemminflect.getAllInflections(lemma).values() for infl in tup if infl != word])), lemma)
                    else:
                        df = inflecteur.dico_transformer.loc[(inflecteur.dico_transformer.lemma == lemma) & (inflecteur.dico_transformer.index!=word)].reset_index(drop=False)
                        inflections = (i, df[df['gram'] == word_tag].part.unique().tolist(), lemma)

                random.shuffle(inflections[1])
                token_inflections.append(inflections)
        return token_inflections
    
    def get_inflections_multi(self, orig_tokenized, pos_tagged, constrain_pos):
        have_inflections = {'NOUN', 'VERB', 'ADJ'}
        universal_to_infelect = {'NOUN': 'Nom', 'VERB': 'Verbe', 'ADJ': 'Adjectif'}
        token_inflections = [] # elements of form (i, inflections) where i is the token's position in the sequence

        for i, word in enumerate(orig_tokenized):
            lemmas = self.get_lemmas(word)
            if lemmas and pos_tagged[i][1] in have_inflections:
                word_tag = pos_tagged[i][1] if self.lang=='en' else universal_to_infelect[pos_tagged[i][1]]
                if word_tag in lemmas:
                    lemma = lemmas[word_tag][0]
                else:
                    lemma = random.choice(list(lemmas.values()))[0]

                if constrain_pos:
                    if self.lang=='en':
                        inflections = (i, list(set([infl for tup in lemminflect.getAllInflections(lemma, upos=pos_tagged[i][1]).values() for infl in tup])), lemma)
                    else:
                        df = inflecteur.dico_transformer.loc[(inflecteur.dico_transformer.lemma == lemma)].reset_index(drop=False)
                        inflections = (i, df[df['gram'] == word_tag].part.unique().tolist(), lemma)
                else:
                    if self.lang=='en':
                        inflections = (i, list(set([infl for tup in lemminflect.getAllInflections(lemma).values() for infl in tup])), lemma)
                    else:
                        df = inflecteur.dico_transformer.loc[(inflecteur.dico_transformer.lemma == lemma)].reset_index(drop=False)
                        inflections = (i, df[df['gram'] == word_tag].part.unique().tolist(), lemma)
                        
                random.shuffle(inflections[1])
                token_inflections.append(inflections)
        return token_inflections

    def morph(self, source, reference, constrain_pos=True, multi=False):
        doc = self.nlp(source)
        orig_tokenized = [token.text for token in doc]
        spaces = [token.whitespace_ for token in doc]
        tags = [token.tag_ if self.lang=='en' else token._.melt_tagger for token in doc]
        pos_tagged = [(token, map_tag(self.lang, tag))
                      for (token, tag) in zip(orig_tokenized, tags)]
        pos_tagged = [(tagged[0], '.') if '&' in tagged[0] else tagged for tagged in pos_tagged]
        
        token_inflections = self.get_inflections_multi(orig_tokenized, pos_tagged, constrain_pos)

        original_score, orig_predicted = self.get_scores([orig_tokenized], reference)
        original_score, orig_predicted = original_score[0], orig_predicted[0]

        if multi:
            forward_perturbed, final_token_idx, final_infl, final_lemma, forward_score, \
            forward_predicted, num_queries_forward = self.search_seq2seq_multi(token_inflections,
                                                                        orig_tokenized,
                                                                        original_score,
                                                                        reference)
        else:
            forward_perturbed, final_token_idx, final_infl, final_lemma, forward_score, \
            forward_predicted, num_queries_forward = self.search_seq2seq(token_inflections,
                                                                        orig_tokenized,
                                                                        original_score,
                                                                        reference)
        # If the score is the same we leave the source as is
        if forward_score >= original_score:
            # print("No perturbation found: ", source, token_inflections, forward_score, original_score)
            forward_perturbed = orig_tokenized
            forward_predicted = orig_predicted
            final_token_idx = None
            final_infl = None
            final_lemma = None

        num_queries = num_queries_forward + 1
        
        detokenized_perturbed = spacy.tokens.Doc(self.nlp.vocab, words=forward_perturbed, spaces=spaces).text
        return forward_perturbed, detokenized_perturbed, final_token_idx, final_infl, final_lemma, forward_predicted, num_queries
    
    def morph_sentences(self, sentences, references, constrain_pos=True, multi=False):
        perturbed = []
        detokenized_perturbed = []
        inflection_idx = []
        inflection = []
        label_ids = []
        labels = []
        preds = []
        queries = []
        lemmas = []
        for source, reference in tqdm.tqdm(zip(sentences, references)):
            curr_perturbed, curr_detokenized_perturbed, curr_token, curr_infl, curr_lemma, curr_predicted, curr_queries = self.morph(source, reference, constrain_pos, multi)
            
            curr_label_ids = [0] * len(curr_perturbed)
            if curr_token is not None:
                if isinstance(curr_token, list):
                    for idx in curr_token:
                        curr_label_ids[idx] = 1
                else:
                    curr_label_ids[curr_token] = 1
            perturbed.append(curr_perturbed)
            detokenized_perturbed.append(curr_detokenized_perturbed)
            inflection_idx.append(curr_token)
            inflection.append(curr_infl)
            lemmas.append(curr_lemma)
            label_ids.append(curr_label_ids)
            labels.append('noisy' if curr_token is not None else 'clean')
            preds.append(curr_predicted)
            queries.append(curr_queries)
        return perturbed, detokenized_perturbed, inflection_idx, inflection, lemmas, label_ids, labels, preds, queries
        

    def search_seq2seq(self, token_inflections, orig_tokenized,
                       original_score, reference):
        num_queries = 0
        max_predicted = ''
        
        curr_token = next(iter(token_inflections), None)
        
        if curr_token is None:
            return orig_tokenized, None, None, None, original_score, max_predicted, num_queries
        
        all_perturbed = []
        all_inflections = []
        all_inflection_idx = []
        all_lemmas = []
        for curr_token in token_inflections:
            for infl in curr_token[1]:
                perturbed_tokenized = orig_tokenized.copy()
                perturbed_tokenized[curr_token[0]] = infl
                all_perturbed.append(perturbed_tokenized)
                all_inflections.append(infl)
                all_inflection_idx.append(curr_token[0])
                all_lemmas.append(curr_token[2])
        
        if len(all_perturbed)==0:
            return orig_tokenized, None, None, None, original_score, max_predicted, num_queries
        
        all_scores, predicted = self.get_scores(all_perturbed, reference)
        min_score_idx = np.argmin(all_scores)
        min_score = all_scores[min_score_idx]
        perturbed_tokenized = all_perturbed[min_score_idx]
        max_infl = all_inflections[min_score_idx]
        max_infl_token_idx = all_inflection_idx[min_score_idx]
        lemma = all_lemmas[min_score_idx]
        max_predicted = predicted[min_score_idx]
        num_queries = len(all_scores)
        
        assert len(all_scores) == len(all_perturbed) == len(all_inflections) == len(all_inflection_idx)
        assert max_infl==perturbed_tokenized[max_infl_token_idx]
        
        return perturbed_tokenized, max_infl_token_idx, max_infl, lemma, min_score, max_predicted, num_queries
    
    
    def search_seq2seq_multi(self, token_inflections, orig_tokenized,
                       original_score, reference):
        num_queries = 0
        max_predicted = ''
        
        curr_token = next(iter(token_inflections), None)
        
        if curr_token is None:
            return orig_tokenized, None, None, None, original_score, max_predicted, num_queries
        
        final_lemma = []
        final_max_infl = []
        final_max_infl_token_idx = []
        perturbed_tokenized = orig_tokenized.copy()
        for curr_token in token_inflections:
            all_perturbed = []
            all_inflections = []
            all_inflection_idx = []
            all_lemmas = []
            for infl in curr_token[1]:
                perturbed_tokenized[curr_token[0]] = infl
                all_perturbed.append(perturbed_tokenized)
                all_inflections.append(infl)
                all_inflection_idx.append(curr_token[0])
                all_lemmas.append(curr_token[2])

            if len(all_perturbed)==0:
                continue
            all_scores, predicted = self.get_scores(all_perturbed, reference)
            num_queries += len(all_scores)
            min_score_idx = np.argmin(all_scores)
            min_score = all_scores[min_score_idx]
            perturbed_tokenized[curr_token[0]] = all_inflections[min_score_idx]
            if all_inflections[min_score_idx] == orig_tokenized[curr_token[0]]:
                continue
            final_max_infl.append(all_inflections[min_score_idx])
            final_max_infl_token_idx.append(all_inflection_idx[min_score_idx])
            final_lemma.append(all_lemmas[min_score_idx])
            max_predicted = predicted[min_score_idx]
        
        return perturbed_tokenized, final_max_infl_token_idx, final_max_infl, final_lemma, min_score, max_predicted, num_queries
    

class MorpheusNMT(MorpheusSeq2Seq):
    def __init__(self):
        super().__init__()

    def get_scores(self, source, reference):
        predicted = self.model_predict(source)
        scores = [sacrebleu.sentence_bleu(p, [reference]).score for p in predicted]
        return scores, predicted

    @abstractmethod
    def model_predict(self, source, **kwargs):
        pass


class MorpheusHuggingfaceNMT(MorpheusNMT):
    def __init__(self, model_name, max_input_tokens=1024, use_cuda=True, src_lang='en', tgt_lang=None, multilingual=False, batch_size=32):
        if torch.cuda.is_available() and use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model_name = model_name
        config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name,config=config)
        self.model.eval()
        self.model.to(self.device)
        self.max_input_tokens = max_input_tokens
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.multilingual = multilingual
        self.batch_size = batch_size
        super().__init__()

    def map_lang(self, lang):
        try:
            with open("./langcodes.json", "r") as f:
                langcodes = json.load(f)
        except FileNotFoundError:
            print("language mapping file not found")
            return lang
        model_langcodes = langcodes.get(self.model_name)
        lang_token = model_langcodes.get(lang) if model_langcodes is not None else None
        return lang_token
        
    
    def model_predict(self, source):
        
        if self.multilingual:
            src_token = self.map_lang(self.src_lang)
            tgt_token = self.map_lang(self.tgt_lang)
            self.tokenizer.src_lang = src_token.strip("__")
        
        predicted = []
        for i in range(0, len(source), self.batch_size):
            source_batch = source[i:i+self.batch_size]
            tokenized = self.tokenizer(source_batch, max_length=self.max_input_tokens, 
                                                    return_tensors='pt', is_split_into_words=True, 
                                                    truncation=True, padding=True).to(self.device)
            if not self.multilingual:
                generated = self.model.generate(**tokenized)
            else:
                generated = self.model.generate(**tokenized, forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_token))
            predicted.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
            
        return predicted

