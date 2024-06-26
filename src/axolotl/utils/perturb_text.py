import json
import spacy
import random
import tarfile
import unicodedata
import os
import re

import numpy as np
from unidecode import unidecode
from nltk.corpus import words

def is_special_token(token):
    return token.startswith('<|') and token.endswith('|>') and token.count(' ') == 0

def get_words_and_separators(nlp, text):
    if '<|' in text and '|>' in text:
        # A faster check to see if regexp split is necessary, which can be slow.
        special_token_split = r'(<\|[^\s<|>]*\|>)'
        special_token_parts = re.split(special_token_split, text)
        special_token_parts = [p for p in special_token_parts if len(p) != 0]
    else:
        special_token_parts = [text]
    all_tokens, all_separators = [], []
    start = 0
    for part_idx, part in enumerate(special_token_parts):
        if is_special_token(part):
            tokens = [part]
        else:
            tokens = [token.text for token in nlp(part)]
        separators = []
        for token in tokens:
            # Find the position of the next token in the string, starting from 'start'
            pos = text.find(token, start)
            assert pos != -1, print(f'{text} \n {tokens} \n violating {start}: {token}')
            # Extract the separator: the part of the string before this token
            if start != 0:
                # Very first token of text does not have separator.
                separator = text[start:pos]
                separators.append(separator)
            # Update 'start' to the end of the token
            start = pos + len(token)
        if part_idx == 0:
            # Very first token of text does not have separator.
            assert len(separators) == len(tokens) - 1, f'{part}. Tokens: {tokens}, Seps: {separators}'
        else:
            assert len(separators) == len(tokens), f'{part}. Tokens: {tokens}, Seps: {separators}'
        all_tokens += tokens
        all_separators += separators
    # Add the trailing separator, if any
    all_separators.append(text[start:])
    assert len(all_separators) == len(all_tokens), f'{text}. Tokens: {all_tokens}, Seps: {all_separators}'
    return all_tokens, all_separators

def transfer_capitalization(source_word, target_word):
    if source_word.istitle():  # If the source word is title cased (capitalized)
        return target_word.title()
    elif source_word.isupper():  # If the source word is upper cased
        return target_word.upper()
    elif source_word.islower():  # If the source word is lower cased
        return target_word.lower()
    else:  # If the source word has a mix of upper and lower case letters
        # Transfer the capitalization of each character
        return ''.join(t.upper() if s.isupper() else t.lower() for s, t in zip(source_word, target_word.ljust(len(source_word))))

def is_punctuation(s):
    for char in s:
        if unicodedata.category(char).startswith('P'):
            continue
        else:
            return False
    return True

newline_chars = set(['LINE FEED (LF)', 'VERTICAL TABULATION', 'FORM FEED (FF)', 'CARRIAGE RETURN (CR)',
                     'NEXT LINE (NEL)', 'LINE SEPARATOR', 'PARAGRAPH SEPARATOR'])
def is_newline(s):
    # List of newline-like characters' names in Unicode
    # Get the name of the character
    for char in s:
        try:
            char_name = unicodedata.name(char)
        except ValueError:
            return False   # character not found in Unicode database
        if char_name not in newline_chars:
            return False
    return True

HOMOPHONE_FILE = './src/axolotl/utils/filtered_dist_homophones.tar.gz'

class Perturber:
    _instance = None

    @classmethod
    def instance(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = cls(*args, **kwargs)
        return cls._instance

    def __init__(self):
        print('Creating Perturber')
        self.disable = False
        self.nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner', 'lemmatizer'])
        self.nlp.max_length = 2000000
        self.homophones = {}
        assert os.path.isfile(HOMOPHONE_FILE), HOMOPHONE_FILE
        with tarfile.open(HOMOPHONE_FILE, 'r:gz') as tar:
            with tar.extractfile('filtered_dist_homophones.jsonl') as f:
                for l in f.readlines():
                    self.homophones |= json.loads(l.strip())
        self.perturb_schedule = list(np.arange(0.12, 0.01, -0.02))
        self.deletion_prob = 0.3
        self.repeat_prob = 0.2
        self.bad_random_prob = 0.02

    def normalize_for_lookup(self, word):
        return unidecode(word.lower())

    def perturb_text(self, text, perturb_prob_factor=1.0, skip_perterb_prob=999):
        if self.disable:
            return text
        if len(text) >= (self.nlp.max_length - 1):
            return text
        if skip_perterb_prob < 1 and random.random() <= skip_perterb_prob:
            return text
        if text == '':
            return text
        
        tokens, separators = get_words_and_separators(self.nlp, text)
        new_tokens, new_separators = [], []
        perturb_idx = 0
        for idx in range(len(tokens)):
            if is_punctuation(tokens[idx]) or is_special_token(tokens[idx]):
                perturb = False
            else:
                perturb_prob = self.perturb_schedule[perturb_idx] if perturb_idx < len(self.perturb_schedule) else self.perturb_schedule[-1]
                perturb_idx += 1
                perturb = random.random() <= (perturb_prob * perturb_prob_factor)
            
            if perturb:
                perturb_type_dice = random.random()
                if perturb_type_dice <= self.deletion_prob:
                    if idx > 0 and separators[idx-1] == '':
                        separators[idx-1] = ' '
                    pass  # Delete token (and separator)
                elif perturb_type_dice <= self.deletion_prob + self.repeat_prob:
                    # Repetition error
                    new_tokens.append(tokens[idx])
                    new_separators.append(' ')
                    new_tokens.append(tokens[idx][1:] if tokens[idx].startswith('\'') else tokens[idx])
                    new_separators.append(separators[idx])
                elif perturb_type_dice <= self.deletion_prob + self.repeat_prob + self.bad_random_prob:
                    new_tokens.append(tokens[idx])
                    new_separators.append(' ')
                    random_word = random.choice(words.words())
                    new_tokens.append(random_word[1:] if random_word.startswith('\'') else random_word)
                    new_separators.append(separators[idx])
                else:
                    # Substitution
                    homophones = self.homophones.get(self.normalize_for_lookup(tokens[idx]))
                    perturbed = tokens[idx]
                    if homophones:
                        perturbed = transfer_capitalization(perturbed, random.choice(homophones))
                    else:
                        perturb_idx -= 1
                    new_tokens.append(perturbed)
                    new_separators.append(separators[idx])
            else:
                new_tokens.append(tokens[idx])
                new_separators.append(separators[idx])

            # Reset perturb schedule if it looks like a new sentence.
            MIN_WORDS_PER_SENTENCE = 20
            if is_punctuation(tokens[idx]) and is_newline(separators[idx]) and perturb_idx >= MIN_WORDS_PER_SENTENCE:
                perturb_idx = 0

        assert len(new_tokens) == len(new_separators)
        if len(new_tokens) == 0:
            return text  # Fallback
        output = [new_tokens[idx] + new_separators[idx] for idx in range(len(new_tokens) - 1)]
        output_str = ''.join(output) + new_tokens[-1]
        # Sanity
        if len(text) >= 128:
            if len(output_str) >= 2 * len(text):
                return text
        #print(f'Perturbed: {text} -> {output_str}')
        return output_str
