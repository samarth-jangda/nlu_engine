# following script is used to process the NER tags o the live predicted chunks
# by the ASR model

import pandas as pd
import spacy
from difflib import SequenceMatcher

open_csv_file=pd.read_csv("/home/samarth/testing/extra_testing/local_salesphony_testing/salesphony_testing/vox/entities.csv")
global open_transcript_file

def process_data(file_path):
    """
    Following function is used to:
    1) Import the model and tokenizer
    2) process the chunk to the model and get predicted tag  
    """
    entities={}
    final_etys={}
    question_ety=[]
    open_transcript_file=open(file_path,'r').readlines()
    while ("\n" in open_transcript_file):
        open_transcript_file.remove("\n")
    for sentence in open_transcript_file:
        predicted_domain_tag=[predict_domain_tag(word,open_transcript_file,open_transcript_file.index(sentence)) for idx,word in enumerate(sentence.replace("\n","").split(" "))] # ----------> json of word and corresponding entity
        question_ety.append(predicted_domain_tag)
    ent_list=[item for sublist in question_ety for item in sublist]    
    while (None in ent_list):
        ent_list.remove(None)
    for item in ent_list:
        for key,value in item.items():
            if key in entities and value!=None:
                entities[key] +=' '+value
            elif value!=None:
                entities[key]=value
    fetch_keys=list(entities)            
    for x in fetch_keys:
        ety_value=entities.get(x)
        remove_duplicates=" ".join(list(set(ety_value.split(" "))))
        get_first_value=remove_duplicates.split(" ")[0] if len(remove_duplicates)==1 else remove_duplicates
        final_etys[x]=get_first_value
    # open_save_json=open("/home/samarthjangda/testing/salesphony_testing/entities.json",'w')
    # json.dump(entities,open_save_json,indent=6)
    # open_save_json.close()
    key_similarity=check_key_similarity(final_etys)
    return(final_etys)

def check_key_similarity(json_data):
    """
    The following function uses spacy hindi model 
    to check if 2 sentences as key in json are similar
    """
    nlp=spacy.blank("hi")
    fetch_keys=list(json_data)
    for key in range(0,len(fetch_keys)+1):
        for data in json_data:
            sent_1=nlp(fetch_keys[key])
            sent_2=nlp(data)
            check_similarity=sent_1.similarity(sent_2)
            print(check_similarity)
            
         

def predict_domain_tag(word,transcripts,sentence_idx):
    """
    Following is a supportive function to fetch
    only the domain specific tags from a transcript 
    """ 
    ety_pair={}
    value=8
    threshold=0.329
    trans=" ".join(transcripts).split(" ")
    for question,entity_list in zip(open_csv_file["question"],open_csv_file["entity"]):
        if word in entity_list.split(","):
            ety_pair[question]=word
    if len(ety_pair) ==1:
        first_quest=next(iter(ety_pair))
        # joined_words=" ".join(transcripts[sentence_idx-value:sentence_idx]).replace("\n","")
        word_index=' '.join(transcripts).replace("\n","").split(" ").index(word)
        prev_trans=' '.join(' '.join(transcripts).replace("\n","").split(" ")[word_index-value:word_index])
        similarity=SequenceMatcher(None,prev_trans,first_quest).ratio()
        print(similarity)
        return{first_quest:ety_pair.get(first_quest)} if similarity > threshold else print("No entitiy found")
    elif len(ety_pair) > 1:
        # joined_words=" ".join(transcripts[sentence_idx-1:sentence_idx])
        word_index=' '.join(transcripts).replace("\n","").split(" ").index(word)
        joined_words=' '.join(' '.join(transcripts).replace("\n","").split(" ")[word_index-value:word_index])
        # word_index=joined_words.split(" ").index(word)
        # words_to_match=" ".join(joined_words.split(" ")[word_index-value:word_index]) if word_index > value else " ".join(joined_words.split(" ")[0:word_index])
        find_smilarity=[SequenceMatcher(None,joined_words,item).ratio() for item in ety_pair]
        max_value_idx=find_smilarity.index(max(find_smilarity))
        counter=0
        extract_value=None
        if max_value_idx !=0 and find_smilarity[max_value_idx] > threshold:
            for key in ety_pair:
                if counter==max_value_idx:
                    extract_value=key 
                counter+=1
            return {extract_value:word}  
        elif max_value_idx==0 and find_smilarity[max_value_idx]>threshold:
            first_quest=next(iter(ety_pair))
            return {first_quest:word}
        # else:
        #     return("None")
        # trans_similarity=similarity_check(transcripts[sentence_idx])
        # old_trans_similarity=similarity_check(transcripts[sentence_idx-1])
        # print(trans_similarity,old_trans_similarity)
    
# sentence=process_data("कितना साइज प्लैन कर रहे हो वनप्लस टू बीएचके")
transcript_file=process_data("/home/samarth/testing/extra_testing/local_salesphony_testing/salesphony_testing/prediction.txt")
#  अच्छा कमर्शियल मादकता हो ठीक है तो आप रागी टू मॉर्गन देख रहे हो या अंडर कंस्ट्रक्शन देखा
# अच्छा और अपनी लोकेशन जायेगी दिल्ली गुड़गांव हाँ दो तीन 