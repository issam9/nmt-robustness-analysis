
from nltk.tag.mapping import map_tag as map_tag_en
import pandas as pd
import string


lefff_to_universal = {
    "ADJ": "ADJ",
    "ADJWH": "ADJWH",
    "ADV": "ADV",
    "ADVWH": "ADV",
    "CC": "CONJ",
    "CLO": "PRON",
    "CLR": "PRON",
    "CLS": "PRON",
    "CS": "CONJ",
    "DET": "DET",
    "DETWH": "DET",
    "ET": "X",
    "I": "INTJ",
    "NC": "NOUN",
    "NPP": "NOUN",
    "P": "ADP",
    "P+D": "ADP",
    "P+PRO": "ADP",
    "PONCT": ".",
    "PREF": "X",
    "PRO": "PRON",
    "PROREL": "PRON",
    "PROWH": "PRON",
    "V": "VERB",
    "VIMP": "VERB",
    "VINF": "VERB",
    "VPP": "VERB",
    "VPR": "VERB",
    "VS": "VERB",
}


lefff_to_inflecteur = {
    "ADJ": "Adjectif",
    "ADJWH": "Adjectif interrogatif", 
    "ADV": "Adverbe",
    "ADVWH": "Adverbe",
    "CC": "Conjonction de coordination",
    "CLO": "Pronom",
    "CLR": "Pronom",
    "CLS": "Pronom",
    "CS": "Conjonction de subordination",
    "DET": "Déterminant",
    "DETWH": "Déterminant",
    "ET": "Partie de composé", 
    "I": "Interjection",
    "NC": "Nom",
    "NPP": "Nom",
    "P": "Préposition",
    "P+D": "Partie de composé", 
    "P+PRO": "Partie de composé",  
    "PONCT": "PONCT",
    "PREF": "Préfixe",
    "PRO": "Pronom",
    "PROREL": "Pronom",
    "PROWH": "Pronom",
    "V": "Verbe",
    "VIMP": "Verbe",
    "VINF": "Verbe",
    "VPP": "Verbe",
    "VPR": "Verbe",
    "VS": "Verbe",
}


def map_tag(lang, tag):
        if lang == 'en':
            return map_tag_en("en-ptb", 'universal', tag)
        elif lang == 'fr':
            return lefff_to_universal.get(tag, "X")
        else:
            raise ValueError("Language not supported")
        
def get_fr_lemmas(word, inflecteur):
    try:
        df = inflecteur.dico_transformer.loc[word]
    except KeyError:
        return {}

    formatted_data = {}

    if isinstance(df, pd.Series): 
        gram = df['gram']
        lemma = df['lemma']
        formatted_data[gram] = (lemma,)
    elif isinstance(df, pd.DataFrame): 
        for gram in df['gram'].unique():
            lemmas = tuple(df[df['gram'] == gram].lemma.tolist())
            formatted_data[gram] = lemmas
    else:
        return {}

    return formatted_data


def get_forme(x, inflecteur):
    word_df = inflecteur.dico_transformer.loc[x['word']]
    if isinstance(word_df, pd.Series):
        word_df = word_df.to_frame().T
    forme_df = word_df.loc[(word_df['gram']==lefff_to_inflecteur.get(x['inflected_pos'], 'X'))]
    if len(forme_df) == 0:
        return None
    if len(forme_df) > 1:
        forme_df = forme_df.loc[forme_df['lemma'] == x['lemma']]
        print(f"Multiple forms found for {x['word']}: {x['lemma']}")
    return ":".join(forme_df.forme.tolist())


def get_forme_multi(x, inflecteur):
    all_forme = []
    for word, lemma, pos in zip(x['word'], x['lemma'], x['inflected_pos']):
        word_df = inflecteur.dico_transformer.loc[word]
        if isinstance(word_df, pd.Series):
            word_df = word_df.to_frame().T
        forme_df = word_df.loc[(word_df['gram']==lefff_to_inflecteur.get(pos, 'X'))]
        if len(forme_df) == 0:
            return None
        if len(forme_df) > 1:
            forme_df = forme_df.loc[forme_df['lemma'] == lemma]
            print(f"Multiple forms found for {word}: {lemma}")
            
        all_forme.extend(forme_df.forme.tolist())
    return ":".join(all_forme)


def get_punctuated_words(words):
    punct_words = []
    for word in words:
        """Checks if a word contains any punctuation characters."""
        for char in word:
            if char in string.punctuation:
                punct_words.append(word)
    return punct_words