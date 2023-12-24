import re
import math
import json
import pandas as pd
import string 

# common phrases in legal documents
re_thuchientheo = re.compile(
    r"((((được\s)?thực hiện theo qu[iy] định tại\s|hướng dẫn tại\s|theo qu[iy] định tại\s|(được\s)?thực hiện theo\s|theo qu[iy] định tại\s|theo nội dung qu[yi] định tại\s|quy[iy] định tại|theo\s)(các\s)?)?|tại\s(các\s)?)(khoản(\ssố)?\s(\d+\,\s)*\d+|điều(\ssố)?\s(\d+\,\s)*\d+|điểm\s(([a-z]|đ)\,\s)*([a-z]|đ)\b|chương(\ssố)?\s(\d+\,\s)*\d+)((\s|\,\s|\s\,\s|\svà\s)(khoản(\ssố)?\s(\d+\,\s)*\d+|điều(\ssố)?\s(\d+\,\s)*\d+|điểm\s(([a-z]|đ)\,\s)*([a-z]|đ)\b|chương(\ssố)?\s(\d+\,\s)*\d+))*(\s(điều này|thông tư này|nghị quyết này|quyết định này|nghị định này|văn bản này|quyết định này))?"
)
re_thongtuso = re.compile(
    r"(thông tư liên tịch|thông tư|nghị quyết|quyết định|nghị định|văn bản|Thông tư liên tịch|Thông tư|Nghị quyết|Nghị định|Văn bản|Quyết định)\s(số\s)?(([a-z0-9]|đ|\-)+\/([a-z0-9]|đ|\-|\/)*)"
)
re_ngay = re.compile(r"ngày\s\d+\/\d+\/\d+\b|ngày\s\d+tháng\d+năm\d+")
re_thang_nam = re.compile(r"tháng\s\d+\/\d+|tháng\s\d+|năm\s\d+")
re_chuong = re.compile(
    r"chương\s(III|II|IV|IX|VIII|VII|VI|XIII|XII|XI|XIV|XIX|XVIII|XVII|XVI|XV|XX|V|X|I|XXIII|XXII|XXI|XXIV|XXVIII|XXVII|XXVI|XXV|XXIX|XXX)\b"
)

# common end phrases in questions
END_PHRASES = [
    "có đúng không",
    "đúng không",
    "được không",
    "hay không",
    "được hiểu thế nào",
    "được quy định cụ thể là gì",
    "được quy định như thế nào",
    "được quy định thế nào",
    "được quy định như nào",
    "trong trường hợp như nào",
    "trong trường hợp như thế nào",
    "trong trường hợp nào",
    "trong những trường hợp nào",
    "được hiểu như thế nào",
    "được hiểu như nào",
    "như thế nào",
    "thế nào",
    "như nào",
    "là gì",
    "là ai",
    "là bao nhiêu",
    "bao nhiêu",
    "trước bao lâu",
    "là bao lâu",
    "bao lâu",
    "bao gồm gì",
    "không",
    "bao gồm những gì",
    "vào thời điểm nào",
    "gồm những giấy tờ gì",
    "những yêu cầu nào",
]

# punctuations, characters, stop-words 
punc = """!"#$%&'()*+,-./:;<=>?@[\]^`{|}~"""  # noqa: W605
table = str.maketrans("", "", punc)

punctuation = [x for x in string.punctuation]
number = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
chars = ["a", "b", "c", "d", "đ", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o"]
stop_word = number + chars + ["của", "và", "các", "có", "được", "theo", "tại", "trong", "về", 
            "hoặc", "người",  "này", "khoản", "cho", "không", "từ", "phải", 
            "ngày", "việc", "sau",  "để",  "đến", "bộ",  "với", "là", "năm", 
            "khi", "số", "trên", "khác", "đã", "thì", "thuộc", "điểm", "đồng",
            "do", "một", "bị", "vào", "lại", "ở", "nếu", "làm", "đây", 
            "như", "đó", "mà", "nơi", "”", "“"]
bm25_removed = punctuation + stop_word

# defining sub-functions

def remove_dieu_number(text):
    '''
    This funtion removes the common legal phrases out from texts
    '''
    text = re_thuchientheo.sub(" ", text)
    text = re_thongtuso.sub(" ", text)
    text = re_ngay.sub(" ", text)
    text = re_thang_nam.sub(" ", text)
    text = re_chuong.sub(" ", text)
    return " ".join(text.split())


def remove_other_number_by_zero(text):
    '''
    This funtion replaces numeric characters in texts into 0 for easier handling
    '''
    for digit in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
        text = text.replace(digit, "0")
    return text


def remove_punct(text):
    '''
    This funtion replaces punctuations in texts for easier handling
    '''
    text = text.replace(";", ",").replace(":", ".").replace("“", " ").replace("”", " ")
    text = "".join(
        [
            c
            if c.isalpha() or c.isdigit() or c in [" ", ",", "(", ")", ".", "/", "-"]
            else " "
            for c in text
        ]
    )
    text = " ".join(text.split())
    return text

def lower_or_keep(text):
    "This funtion lower words but not for abbreviations"
    lst = text.split(" ")
    newlst = [x if x.isupper() else x.lower() for x in lst]
    return " ".join(newlst)

def preprocess_all_title(article_title):
    """
    Preprocess titles of documents
    """
    article_title = lower_or_keep(article_title)
    lst = article_title.split()
    new_lst = []
    for i in range(len(lst)):
        if lst[i] == 'số' and i ==  len(lst)-1:
            new_lst.append(lst[i])
        elif lst[i] == 'số' and "/" in lst[i+1]:
            pass
        elif "/" in lst[i]:
            pass
        else:
            new_lst.append(lst[i])
    article_title = " ".join(new_lst)
    article_title = remove_dieu_number(article_title)
    #article_title = remove_other_number_by_zero(article_title)
    article_title = remove_punct(article_title)
    article_title = article_title.replace("về", "")
    if "do" in article_title and "ban hành" in article_title:
        idx = article_title.rfind("do")
        article_title = article_title[:(idx-1)]
    
    re_head = re.compile(r"(thông tư liên tịch|thông tư|nghị quyết|quyết định|nghị định|văn bản)\s(quy định|hướng dẫn)?")
    article_title = re_head.sub(" ", article_title)
    article_title = article_title.replace("   ", " ")
    article_title = article_title.replace("  ", " ")
    return article_title.strip()

def preprocess_article_title(article_title):
    """
    Preprocess titles of documents
    """
    article_title = lower_or_keep(article_title)
    article_title = " ".join(article_title.split()[2:])  # Dieu 1.
    article_title = remove_dieu_number(article_title)
    #article_title = remove_other_number_by_zero(article_title)
    article_title = remove_punct(article_title)
    return article_title
    
def preprocess_khoan(khoan):
    """
    Perprocess parts in a legal documents
    """
    khoan = lower_or_keep(khoan)
    khoan = khoan.replace("\xa0", "")
    matched = re.match(r"^\d+\.(\d+\.?)?\s", khoan)  # 1. 2.2. 2.2
    if matched is not None:
        khoan = khoan[matched.span()[1]:].strip()

    else:
        matched2 = re.match(r"^[\wđ]\)\s", khoan)
        if matched2 is not None:
            khoan = khoan[matched2.span()[1]:].strip()

    khoan = remove_dieu_number(khoan)
    #khoan = khoan.replace("đ)","")
    khoan = re.sub(r"[\wđ]\) ","", khoan)
    khoan = re.sub(r"[\wđ]\. ","", khoan)
    khoan = re.sub(r"\d+\.\d+\.\d+\. ", "", khoan)
    khoan = re.sub(r"\d+\.\d+\. ", "", khoan)
    khoan = re.sub(r"\d+\. ", "", khoan)
    #khoan = re.sub(r"[0-9]\. ", "", khoan)
    #khoan = remove_other_number_by_zero(khoan)
    khoan = remove_punct(khoan)
    khoan = khoan.replace(". .", ".")
    khoan = khoan.replace("..", ".")
    khoan = khoan.replace(", ,", ",")
    khoan = khoan.replace(",,", ",")
    khoan = khoan.strip()
    return " ".join(khoan.split())


def preprocess_question(q, remove_end_phrase=True):
    """
    Preprocess questions
    """
    q = lower_or_keep(q)
    q = remove_dieu_number(q)
    q = "".join([c if c.isalpha() or c.isdigit() or c == " " else " " for c in q])
    q = remove_punct(q)
    if remove_end_phrase:
        for phrase in END_PHRASES:
            if q.endswith(phrase):
                q = q[: -len(phrase)]
                break

    return q.strip()

'''def tokenise(text, segmenter):
    """
    Segment the texts with vncorenlp-segemnter
    """
    result = segmenter.tokenize(text)
    rlt = ""
    for i in range(len(result)-1):
        rlt += " ".join(result[i])
        rlt += " "
    rlt += " ".join(result[len(result)-1])
    return rlt
'''
def tokenise(text, f):
    """
    Segment the texts with pyvi tokenizer
    """
    return f(text)
    
def remove_stopword(w):
    "Remove stopwords in texts"
    return w not in stop_word

def bm25_process(text, f):
    """
    Processing texts for bm25: remove all puntuations, lower all words
    """
    text = tokenise(text, f)
    words = text.lower().split(" ")
    result = [w for w in words if w not in bm25_removed]
    stripped = " ".join(result)
    result = " ".join(stripped.split(" "))
    return result

def length(sentence):
    "Return the length in words of sentences"
    return len(sentence.split())

def build_corpus(f, corpus_file, law_dict, scorpus_ids, head = False):
    """
    Build a corpus-dataframe
    """
    law_ids = []
    text_ids = []
    article_ids = []
    titles = []
    texts = []
    processed_texts = []
    tokenized_texts = []
    bm25texts = []
    lengths = []
    ids = []
    sub_ids = []
    count = 0

    with open (corpus_file, 'r') as input:
        data = json.load(input)
    
    for law in data:
        for article in law['articles']:
            ids.append(count)
            law_ids.append(law['law_id'])
            article_ids.append(article['article_id'])
            text_id = law['law_id'] + "_" + article['article_id']
            text_ids.append(text_id)
            
            titles.append(article['title'])
            texts.append(article['text'])
            
            title = preprocess_article_title(article["title"])
            head = preprocess_all_title(law_dict[law['law_id']])
            
            cac_khoan = article["text"].split("\n")
            khoan_clean = []
            for khoan in cac_khoan:
                khoan = preprocess_khoan(khoan)
                khoan_clean.append(khoan.strip())
            article_text = " ".join(khoan_clean)
            if head:
                processed_text = head + ". " + title + ". " + article_text
            else:
                processed_text = title + ". " + article_text + ". " + head + "."
            processed_texts.append(processed_text)
            start_sub_id = scorpus_ids.index(count)
            try:
                end_sub_id = scorpus_ids.index(count+1)
                sub_ids.append([i for i in range(start_sub_id, end_sub_id)])
            except:
                sub_ids.append([i for i in range(start_sub_id, len(scorpus_ids))])
            
            try: 
                tokenized_text = tokenise(processed_text, f)
                tokenized_texts.append(tokenized_text)
                lengths.append(length(tokenized_text))
            except:
                tokenized_text = tokenise(processed_text[:50000], f)
                tokenized_texts.append(tokenized_text)
                lengths.append(length(tokenized_text))
            bm25texts.append(bm25_process(processed_text, f))
            count += 1
      
    df = pd.DataFrame()
    df["id"] = ids
    df["law_id"] = law_ids
    df["article_id"] = article_ids
    df["text_id"] = text_ids
    df["title"] = titles
    df["text"] = texts
    df["processed_text"] = processed_texts
    df["sub_id"] = sub_ids
    df["tokenized_text"] = tokenized_texts
    df["bm25text"] = bm25texts
    df["len"] = lengths
    
    return df

def create_sliding_window(tokenized_text, size=200, overlap=64):
    """
    Create list of windows for a text
    """
    sentences = tokenized_text.split(".")
    words = tokenized_text.split(" ")
    title = sentences[0]
    words = [w for w in words if len(w) >0]
    actual_size = size - overlap
    
    windows = []
    n_windows = math.ceil(len(words)/actual_size)
    for i in range(n_windows):
        windows.append(" ".join(words[i*actual_size:i*actual_size + size]))
    for i in range(1, n_windows):
        if not windows[i].startswith("."):
            windows[i] = title + ". " + windows[i]
        else:
            windows[i] = title + windows[i]
    return windows

def build_short_corpus(f, corpus_file, law_dict, head=False, size=200, overlap=64):
    """
    Build a corpus-dataframe
    """
    ids = []
    law_ids = []
    text_ids = []
    article_ids = []
    titles = []
    texts = []
    processed_texts = []
    sub_ids = []
    tokenized_texts = []
    bm25texts = []
    lengths = []

    with open (corpus_file, 'r') as input:
        data = json.load(input)
    idx = 0
    sub_idx = 0
    for law in data:
        for article in law['articles']:
            text_id = law['law_id'] + "_" + article['article_id']
            title = preprocess_article_title(article["title"])
            head = preprocess_all_title(law_dict[law['law_id']])
            cac_khoan = article["text"].split("\n")
            khoan_clean = []
            for khoan in cac_khoan:
                khoan = preprocess_khoan(khoan)
                khoan_clean.append(khoan.strip())
            article_text = " ".join(khoan_clean)
            if head:
                processed_text = head + ". " + title + ". " + article_text
            else:
                processed_text = title + ". " + article_text + ". " + head + "."
            try: 
                tokenized_text = tokenise(processed_text, f)
                tokenized_len = length(tokenized_text)
                if tokenized_len <= size + 10:
                    ids.append(idx)
                    law_ids.append(law['law_id'])
                    article_ids.append(article['article_id'])
                    text_ids.append(text_id)
                    titles.append(article['title'])
                    texts.append(article['text'])
                    processed_texts.append(processed_text)
                    sub_ids.append(sub_idx)
                    tokenized_texts.append(tokenized_text)
                    lengths.append(tokenized_len)
                    bm25texts.append(bm25_process(processed_text, f))
                    sub_idx +=1
                else:
                    windows = create_sliding_window(tokenized_text, size=224, overlap=64)
                    for window in windows:
                        ids.append(idx)
                        law_ids.append(law['law_id'])
                        article_ids.append(article['article_id'])
                        text_ids.append(text_id)
                        titles.append(article['title'])
                        texts.append(article['text'])
                        processed_texts.append(processed_text)
                        sub_ids.append(sub_idx)
                        tokenized_texts.append(window)
                        lengths.append(length(window))
                        bm25texts.append(bm25_process(window, f))  
                        sub_idx +=1
            except:
                actual_size = 50000 - overlap
                big_windows = []
                n_big_windows = math.ceil(len(processed_text)/actual_size)
                for i in range(n_big_windows):
                    big_windows.append("".join(processed_text[i*actual_size:i*actual_size + size]))
                for big_window in big_windows:
                    tokenized_text = tokenise(big_window, f)
                    tokenized_len = length(tokenized_text)
                    if tokenized_len > size + 10:
                        windows = create_sliding_window(tokenized_text, size=224, overlap=64)
                        for window in windows:
                            ids.append(idx)
                            law_ids.append(law['law_id'])
                            article_ids.append(article['article_id'])
                            text_ids.append(text_id)
                            titles.append(article['title'])
                            texts.append(article['text'])
                            processed_texts.append(processed_text)
                            sub_ids.append(sub_idx)
                            tokenized_texts.append(window)
                            lengths.append(length(window))
                            bm25texts.append(bm25_process(window, f))  
                            sub_idx +=1
                    else:
                        ids.append(idx)
                        law_ids.append(law['law_id'])
                        article_ids.append(article['article_id'])
                        text_ids.append(text_id)
                        titles.append(article['title'])
                        texts.append(article['text'])
                        processed_texts.append(processed_text)
                        sub_ids.append(sub_idx)
                        tokenized_texts.append(tokenized_text)
                        lengths.append(tokenized_len)
                        bm25texts.append(bm25_process(processed_text, f))
                        sub_idx +=1
    
            idx += 1
                 
    df = pd.DataFrame()
    df["id"] = ids
    df["law_id"] = law_ids
    df["article_id"] = article_ids
    df["text_id"] = text_ids
    df["title"] = titles
    df["text"] = texts
    df["processed_text"] = processed_texts
    df["sub_id"] = sub_ids
    df["tokenized_text"] = tokenized_texts
    df["bm25text"] = bm25texts
    df["len"] = lengths
    
    return df

def build_qa(f, df, qa_file, split = False):
    """
    Build a question-answer dataframe
    """
    text_ids = df["text_id"].tolist()
    titles = df["title"].tolist()
    texts = df["text"].tolist()
    lengths = df["len"].tolist()
    sub_ids = df["sub_id"].tolist()
    q_texts = []
    q_processed_texts = []
    q_tokenized_texts = []
    q_bm25texts = []
    q_lens = []
    no_ans = []
    ans_ids = []
    ans_text_ids = []
    ans_titles = []
    ans_texts = []
    ans_lens = []
    ans_sub_ids = []
    with open (qa_file, 'r') as input:
        data = json.load(input)
    
    if not split:
        for item in data['items']:
            question = item["question"]
            q_texts.append(question)
            q_processed_text = preprocess_question(question, remove_end_phrase=False)
            q_processed_texts.append(q_processed_text)
            q_tokenized_text = tokenise(q_processed_text, f)
            q_tokenized_texts.append(q_tokenized_text)
            q_bm25texts.append(bm25_process(q_processed_text, f))
            q_lens.append(length(q_tokenized_text))
            ans_text_id = ""
            ans_id = ""
            ans_title = ""
            ans_text = ""
            ans_len = ""
            ans_count = 0
            ans_sub_id = []
            for i in range(len(item['relevant_articles'])):
                ans_count += 1
                atext_id = item['relevant_articles'][i]['law_id'] + "_"  + item['relevant_articles'][i]['article_id']
                a_id = text_ids.index(atext_id)
                ans_text_id += atext_id
                ans_id += str(a_id)
                ans_title += titles[a_id]
                ans_text += texts[a_id]
                ans_len += str(lengths[a_id])
                sub_id = sub_ids[a_id]
                ans_sub_id += sub_id
                
                if i < len(item["relevant_articles"]) - 1:
                    ans_text_id += ", "
                    ans_id += ", "
                    ans_title += ", "
                    ans_text += ", "
                    ans_len += ", "
                    
            no_ans.append(ans_count)
            ans_text_ids.append(ans_text_id)
            ans_ids.append(ans_id)
            ans_titles.append(ans_title)
            ans_texts.append(ans_text)
            ans_lens.append(ans_len)
            ans_sub_ids.append(ans_sub_id)
    else:
        for item in data['items']:
            question = item["question"]
            for article in item['relevant_articles']:
                q_texts.append(question)
                q_processed_text = preprocess_question(question, remove_end_phrase=False)
                q_processed_texts.append(q_processed_text)
                q_tokenized_text = tokenise(q_processed_text, f)
                q_tokenized_texts.append(q_tokenized_text)
                q_bm25texts.append(bm25_process(q_processed_text, f))
                q_lens.append(length(q_tokenized_text))           
                ans_text_id = article['law_id'] + "_"  + article['article_id']
                ans_text_ids.append(ans_text_id)
                a_id = text_ids.index(ans_text_id)
                ans_ids.append(a_id)
                ans_titles.append(titles[a_id])
                ans_texts.append(texts[a_id])
                ans_lens.append(lengths[a_id])
                ans_sub_ids.append(sub_ids[a_id])
                
    
    df = pd.DataFrame()
    df["question"] = q_texts
    df["processed_question"] = q_processed_texts
    df["tokenized_question"] = q_tokenized_texts
    df["bm25_question"] = q_bm25texts
    df["ques_len"] = q_lens
    if not split:
        df['no_ans'] = no_ans
    df["ans_text_id"] = ans_text_ids
    df["ans_id"] = ans_ids
    df["ans_title"] = ans_titles
    df["ans_text"] = ans_texts
    df["ans_len"] = ans_lens
    df["ans_sub_id"] = ans_sub_ids
    
    return df

def build_biencoder_data(dqa_split, bm25, set_ques, no_hneg, no_search):
    """
    Build train, val, test, dataframe used for biencoder training
    """
    qa_ids = []
    neg_ids = []
    search_ids = []
    q_texts = dqa_split['question'].tolist()
    q_bm25texts = dqa_split['bm25_question'].tolist()
    count = 0
    ans_ids = dqa_split['ans_id'].tolist()
    ids = [i for i in range(bm25.corpus_size)]
    for i in range(len(q_texts)):
        if q_texts[i] in set_ques:
            qa_ids.append(i)
            q_bm25 = q_bm25texts[i].split(" ")
            bm25_ids = bm25.get_top_n(q_bm25, ids, n=no_search)
            if ans_ids[i] in bm25_ids:
                count += 1
        
            neg = bm25_ids[:(no_hneg+1)]
            if ans_ids[i] in neg:
                neg.remove(ans_ids[i])
                
            neg = neg[:no_hneg]
            neg_ids.append(neg)
            search_ids.append(bm25_ids)
    print(count/len(qa_ids))   
    df = dqa_split.loc[qa_ids]
    df['neg_ids'] = neg_ids
    df['search_ids'] = search_ids
    return df

def build_short_data(df, dcorpus, limited_length = 234):
    """
    Build short data
    """
    ids = [i for i in range(len(df)) if dcorpus['len'][df['ans_id'][i]] <= limited_length]
    dshort = df.loc[ids].copy(deep= True).reset_index(drop=True)
    return dshort

def build_general_data(dqa, bm25, set_ques, no_hneg, no_search):
    """
    Build general train, test, val dataframe
    """
    qa_ids = []
    neg_ids = []
    search_ids = []
    q_texts = dqa['question'].tolist()
    q_bm25texts = dqa['bm25_question'].tolist()
    ans_ids = dqa['ans_id'].tolist()
    ids = [i for i in range(bm25.corpus_size)]
    count = 0
    
    for i in range(len(q_texts)):
        if q_texts[i] in set_ques:
            qa_ids.append(i)
            q_bm25 = q_bm25texts[i].split(" ")
            ans_id = [int(x) for x in ans_ids[i].split(", ")]
            bm25_ids = bm25.get_top_n(q_bm25, ids, n= no_search)
            search_ids.append(bm25_ids)
            
            for a_id in ans_id:
                if a_id in bm25_ids:
                    bm25_ids.remove(a_id)
            neg_id = bm25_ids[:no_hneg]
            neg_ids.append(neg_id)
            if len(bm25_ids) == (no_search - len(ans_id)):
                count += 1      
        
    df = dqa.loc[qa_ids]
    df['neg_ids'] = neg_ids
    df['search_ids'] = search_ids
    print(count/len(qa_ids))
    return df