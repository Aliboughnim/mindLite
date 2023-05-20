import os
import time
import spacy
import base64
import re

##################
###  Terminal : python -m spacy download en_core_web_lg 
##################

# Load the large English language model into a spaCy pipeline
nlp = spacy.load("en_core_web_lg")  


def create_input_graph(sentence1):
        # Parse sentence:
    #sentence1 = "Paco had 41 cookies and gave 9 cookie .he ate 18 cookis."

    #the conversion of each number written in any letter in the sentence to digits 
    if(detecter_nombre_en_lettres(sentence1)):
        converted_sentence = convert_number_to_digits(sentence1)
        sentence_without_space=additionner_nombres(converted_sentence)
        sentence1=add_space_between_number_and_word(sentence_without_space)

    doc = nlp(sentence1)
    print(doc)

    # Create dependency graph and draw it if there is at least one number
    has_num = False
    for token in doc:
        if token.like_num:
            has_num = True
            break
    dic={}
    text=''
    if has_num:
        #Create a dependency graph from the processed sentence using Spacy
        dic = create_dependency_graph(doc)
        sources=dic['sources']
        targets=dic['targets']
        for i in range(len(sources)):
            text = text+sources[i]+'-'+targets[i]+';'
    return text

###################ok###########################
def detecter_nombre_en_lettres(phrase):
    # Appliquer le modèle Spacy à la phrase donnée
    doc = nlp(phrase)
    
    # Iterate through each token in the document
    for token in doc:
        # Check if the token is labeled as a number
        if token.like_num:
            # Check if the text of the token is a number written in letters
            if token.text.isalpha():
                return True        # Check if the text of the token is a number written in letters

    return False # If no number written in letters is found, return False

#############################
#this function returns the converted sentence with numbers represented as digits
def convert_number_to_digits(sentence):
    # Dictionary to map number words to digits
    number_dict = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
        "thirty": "30",
        "forty": "40",
        "fifty": "50",
        "sixty": "60",
        "seventy": "70",
        "eighty": "80",
        "ninety": "90",
        "hundred": "00"
    }

    converted_sentence = ""
    current_number = ""
   # Loop through each character in the sentence
    for i in range(len(sentence)):
        char = sentence[i]
        # If character is an alphabetical letter, add it to the current number
        if char.isalpha():
            current_number += char
         # If character is a hyphen and the next character is a digit, add the hyphen to the current number
        elif char == '-' and i + 1 < len(sentence) and sentence[i + 1].isdigit():
            current_number += char
        else:
           # If the current number is a valid number word, convert it to a digit and add it to the converted sentence
            if current_number.lower() in number_dict:
                if current_number.lower() == "hundred" and i + 1 < len(sentence) and sentence[i + 1].isalpha():
                    converted_sentence += "00"
                else:
                    converted_sentence += " " + number_dict[current_number.lower()]
                current_number = ""
            else:
                converted_sentence += current_number
                current_number = ""
                converted_sentence += char
 # Check if the last current number is a valid number word, and add it to the converted sentence if it is
    if current_number.lower() in number_dict:
        converted_sentence += " " + number_dict[current_number.lower()]
    else:
        converted_sentence += current_number
    # Remove any double spaces and leading/trailing spaces
    converted_sentence = converted_sentence.replace("  ", " ").strip() # Supprimer les espaces en double et les espaces au début/à la fin
    # Remove any unnecessary zeros (e.g. "02" -> "2")
    converted_sentence = converted_sentence.replace(" 0", " ").replace(" 00", " ") # Supprimer les zéros inutiles

    # Merge adjacent digits to form multi-digit numbers
    words = converted_sentence.split()
    i = 0
    while i < len(words):
        if words[i].isdigit():
            j = i + 1
            while j < len(words) and words[j].isdigit():
                words[i] += words[j]
                words[j] = ""
                j += 1
            i = j
        else:
            i += 1
    
    converted_sentence = " ".join(words)
        
    # Replace missing spaces between adjacent numbers
    converted_sentence = converted_sentence.replace("and", "").replace("  ", " ").strip()

    return converted_sentence

##############################
def additionner_nombres(phrase):
    pattern = r'(\d+)\s+(\d+)'  # Regular expression to detect consecutive integers 
    match = re.search(pattern, phrase)  # Search for the first match of the regex in the sentence
    while match:  # Repeat as long as there are matches
        nombre1 = int(match.group(1))  
        nombre2 = int(match.group(2))  
        somme = nombre1 + nombre2  
        phrase = phrase.replace(match.group(), str(somme))  # Replace the numbers in the sentence with the obtained sum
        match = re.search(pattern, phrase)  # Search for the next match of the regex in the sentence
    return phrase


###############################
def add_space_between_number_and_word(sentence):
    pattern = r'(?<=\d)(?=\D)'
    sentence_with_space = re.sub(pattern, ' ', sentence)
    return sentence_with_space

#########################
###########ok################""

def create_dependency_graph(doc):
    # Initialize an empty list to store edges in the dependency graph
    edges = []
    # Initialize an empty dictionary to combine numerical token
    combined_nodes = {}
    # This function combines a number and its adjacent noun into a single node in the graph.
    def combine_num_noun(token):
         # If the token is a number, combine it with the adjacent noun.
        if token.like_num:
            num_str = token.text
            num_len = len(num_str)
            if num_len == 3:
                combined_nodes[token.i] = num_str
                # Add edges from the combined node to the hundreds, tens, and ones places.
                edges.append((num_str, f"{num_str[0]}00"))
                edges.append((num_str, f"{num_str[1]}0"))
                edges.append((num_str, f"{num_str[2]}"))
            elif num_len == 2:
                combined_nodes[token.i] = num_str
                # Add edges from the combined node to the tens and ones places.
                edges.append((num_str, f"{num_str[0]}0"))
                edges.append((num_str, f"{num_str[1]}"))
            else:
                combined_nodes[token.i] = num_str
            return num_str
        else:
            return token.text


    previous_subject = None
    has_num = False
    # Iterate over each sentence in the spaCy document
    for sent in doc.sents:
        for token in sent:
            # Check if the sentence has a numerical entity
            if token.like_num:
                has_num = True
                break
        # If the sentence does not have a numerical entity, skip it
        if not has_num:
            continue
        subjects = []
        for token in sent:
           # Check if the token is a numerical entity or a subject, auxiliary verb, or root verb
            if token.like_num or token.dep_ in ['nsubj', 'aux', 'ROOT']:
                for child in token.children:
                    # Check if the child is a punctuation mark or a conjunction, or if it is a pronoun
                    if child.pos_ not in ['PUNCT', 'CCONJ'] and child.text.lower() not in ['he', 'she', 'it']:
                        # Add an edge from the numerical entity or subject to the child
                        edges.append((combine_num_noun(token), combine_num_noun(child)))

                 # Check if the head of the token is not the token itself, and if it is not a punctuation mark or a conjunction, or if it is not a pronoun     
                if token.head != token and token.head.pos_ not in ['PUNCT', 'CCONJ'] and token.text.lower() not in ['he', 'she', 'it']:
                    edges.append((combine_num_noun(token.head), combine_num_noun(token)))
                 # If the token is a subject, add it to the list of subjects.    
                if token.dep_ == 'nsubj':
                    if token.text.lower() not in ['he', 'she', 'it']:
                        subjects.append(token)
                   # If the subject is a pronoun, add an edge from the previous subject to the head of the current subject.
                    else:
                        if previous_subject:
                            edges.append((combine_num_noun(previous_subject), combine_num_noun(token.head)))
            elif token.dep_ == 'conj':
                # Find the verb in the conjunction
                verb = None
                for t in token.rights:
                    if t.pos_ == 'VERB':
                        verb = t
                        break
                # If the conjunction has no verb, find the head verb
                if verb is None:
                    verb = token.head
                    while verb.dep_ != 'ROOT' and verb.pos_ != 'VERB':
                        verb = verb.head
                # Add the verb-conjunction edge to the list of edges
                if verb.pos_ not in ['PUNCT']:
                    edges.append((combine_num_noun(verb), combine_num_noun(token)))
                # Add the conjunction's other dependencies to the list of edges
                for child in token.children:
                    if child.like_num or (child.pos_ not in ['PUNCT'] and child.text.lower() not in ['he', 'she', 'it']):
                        edges.append((combine_num_noun(token), combine_num_noun(child)))
                        
     
        # If there is only one subject in the sentence, add it to the list of edges
        if len(subjects) == 1:
            subject = subjects[0]
            previous_subject = subject
            for token in sent:
                if token.dep_ == 'conj' and token.head.pos_ == 'VERB':
                    new_verb = token
                    edges.append((combine_num_noun(subject), combine_num_noun(new_verb)))
        sources=[]
        targets=[]
        # iterate over the edges in the list and add each node to the graph
        for edge in edges:
            # extract the source and target nodes from the edge
            source, target = edge
            sources.append(source)
            targets.append(target)
            
        return {'sources':sources,'targets':targets}