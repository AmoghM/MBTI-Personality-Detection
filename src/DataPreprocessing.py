from cleantext import clean
import pandas as pd
import spacy,re,json
from random import shuffle
nlp = spacy.load('en_core_web_sm')
nlp.max_length = 6500000
import argparse

label_encoding = { "istj":1, "istp":2, "isfj":3, "isfp":4, "infj":5, "infp":6, "intj":7, "intp":8, "estp":9, "estj":10, "esfp":11, "esfj":12, "enfp":13, "enfj":14, "entp":15, "entj":16 }
class DataProcess:
    def __init__(self, path, chunksize=None):
        self.path = path
        self.chunksize = chunksize
        self.dataframe = pd.read_csv(self.path, chunksize=self.chunksize,encoding='utf-8', engine='c')
    
    def clean_text(self, text):
            return clean(text,
            fix_unicode=True,               # fix various unicode errors
            to_ascii=True,                  # transliterate to closest ASCII representation
            lower=True,                     # lowercase text
            no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
            no_urls=True,                  # replace all URLs with a special token
            no_emails=True,                # replace all email addresses with a special token
            no_phone_numbers=True,         # replace all phone numbers with a special token
            no_numbers=True,               # replace all numbers with a special token
            no_digits=True,                # replace all digits with a special token
            no_currency_symbols=True,      # replace all currency symbols with a special token
            no_punct=True,                 # fully remove punctuation
            replace_with_url="<URL>",
            replace_with_email="<EMAIL>",
            replace_with_phone_number="<PHONE>",
            replace_with_number="<NUMBER>",
            replace_with_digit="0",
            replace_with_currency_symbol="<CUR>",
            lang="en"                       # set to 'de' for German special handling
        )
        
    
    @staticmethod
    def fix_clean(text):
        text = re.sub("http.*", ' ',text)
        text = re.sub("<u+.*>", ' ', text)
        text = re.sub(r'[^\w\s<>]','',text)
        return text
    
    @staticmethod
    def fit_clean(df):
        data = []
        count = 0
        for chunk in df:
            for index, row in chunk.iterrows():
                temp = row
                temp['comment'] = DataProcess.fix_clean(str(row['comment']))
                data.append(temp)
                count+=1
                if count%100==0:
                    print(count)
        return data
        
    def sentence_tokenize(self, comment):
        doc = nlp(comment)
        comments=[]
        for i, token in enumerate(doc.sents):
            text = str(token.text)
            if len(text.split())<10:
                continue
            text = self.clean_text(text)
            comments.append(text)
        
        return ' '.join(comments) 
           
    def fit(self):
        self.data = []
        self.error = []
        count = 0
        for chunk in self.dataframe:
            for index, row in chunk.iterrows():
                temp = row
                try:
                    temp['comment'] = self.sentence_tokenize(str(row['comment']))
                    self.data.append(temp)
                except Exception:
                    self.error.append(temp)
                    continue

                if count%self.chunksize==0:
                    print(count/self.chunksize, len(self.data))
                count+=1
    
    @staticmethod
    def seq_df_chunk(dataframe,seq_len=60,breakpoint=None):
        count=0
        data = []
        for chunk in dataframe:
            comments = chunk['comment']
            labels = chunk['type']
            for comment,label in zip(comments,labels):
                comment_split = str(comment).split()
                comment_chunk = [comment_split[i:i + seq_len] for i in range(0, len(comment_split), seq_len)]
                print(comment_chunk)
                break
                for i in comment_chunk:
                    data.append({"comment":i,"type":label_encoding[label]})
                    count+=1
                if count%10000==0:
                    print(count/10000)

        return data
    
    @staticmethod
    def seq_df(df,seq_len=60):
        count=0
        chunks = []
        for index, row in df.iterrows():
            comment = row['comment']
            label = row['type']
            comment_split = str(comment).split()
            comment_chunk = [comment_split[i:i + seq_len] for i in range(0, len(comment_split), seq_len)]
            for i in comment_chunk:
                chunks.append({"comment":' '.join(i),"type":label_encoding[label]})
            
            if count%1000 == 0:
                print(count)
            count+=1
        return chunks
    

    @staticmethod
    def export_data_csv(data,write_path,col,header=False):
        df_data = pd.DataFrame.from_dict(data, orient='columns')
        df_data = df_data[col]
        df_data.to_csv(write_path,index=False,encoding="utf-8",header=header)
        return df_data
        
    @staticmethod
    def export_data_json(data,path):
        with open(path,"a+",encoding="utf-8") as fw:
            for idx, row in enumerate(data):
                if idx%100000==0:
                    print(idx/100000)
                json.dump(row, fw)
                fw.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--read", default="../data/Reddit-19K personality data/mbti9k_comments.csv")
    parser.add_argument("--clean_write",default="../data/mbti_comments/mbti_comments_cleaned.csv")
    parser.add_argument("--write_chunk",default="../data/mbti_comments/extra_clean_seq_60.csv")
    parser.add_argument("--extra_clean",default="../data/mbti_comments/mbti_comments_extra_clean.csv")

    args = parser.parse_args()
    dc = DataProcess(args.read,chunksize=1000)
    dc.fit()
    dc.export_data_csv(dc.data,write_path=args.clean_write,col=["comment","type"])

    df = pd.read_csv(args.clean_write,chunksize=1000)
    data = DataProcess.fit_clean(df)
    df_data = DataProcess.export_data_csv(data,args.extra_clean,["comment","type"])
    df_chunks = DataProcess.seq_df(df_data)
    shuffle(df_chunks)
    DataProcess.export_data_csv(df_chunks,args.write_chunk,['comment','type']).head()


