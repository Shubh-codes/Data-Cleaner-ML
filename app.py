import streamlit as st
import streamlit.components.v1 as stc 
import neattext.functions as nfx

import pandas as pd

import base64
import time 

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from wordcloud import  WordCloud

import spacy
from spacy import  displacy
import wordcloud

import seaborn as sns


nlp = spacy.load("en_core_web_sm")


timestr = time.strftime("%H%M - %d%m%Y")

def plot_wordcloud(raw_text):
    my_wordcloud = WordCloud().generate(raw_text)
    fig = plt.figure()
    plt.imshow(my_wordcloud,interpolation="bilinear")
    plt.axis('off')
    st.pyplot(fig)


def text_analyzer(raw_text):
    docx = nlp(raw_text)
    allData = [(token.text,token.shape_,token.pos_,token.tag_,token.lemma_,token.is_alpha,token.is_punct,token.is_stop) for token in docx]
    df = pd.DataFrame(allData,columns=['Token','Shape','Pos','Tag','Lemma','Is Alpha','Is Punctuation,','Is Stop'])
    return df

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{} </div> """

def get_entities(raw_text):
    docx = nlp(raw_text)
    entities = [(entity.text,entity.label_)for entity in docx.ents]
    return entities
    



def render_entities(raw_text):
	docx = nlp(raw_text)
	html = displacy.render(docx,style='ent')
	html = html.replace("\n\n","\n")
	result = HTML_WRAPPER.format(html)
	stc.html(result,height=3000)

def text_downloader(raw_text):
    b64 = base64.b64encode(raw_text.encode()).decode()
    new_filename = "cleaned_data_{}.txt".format(timestr)
    st.markdown("Download File")
    href = f'<a href="data:file/txt;base64 ,{b64}" download="{new_filename}"> Click Here </a>'
    st.markdown(href,unsafe_allow_html=True)




def make_dowloadable(data):
    csvfile = data.to_csv(index = False)
    b64 = base64.b64encode(csvfile.encode()).decode()
    new_filename = "Nlpified Results{}.csv".format(timestr)
    st.markdown('Download CSV file')
    href = f'<a href="data:file/csv;base64 ,{b64}" download="{new_filename}"> Click Here </a>'
    st.markdown(href,unsafe_allow_html=True)

    
def main():
    st.title("Data Cleaner")

    menu = ['Data Cleaner','About']
    choice = st.sidebar.selectbox('Men',menu)

    if choice == 'Data Cleaner':
        st.subheader('Data Cleaning')

        text_file = st.file_uploader("Upload the data to clean...",type =['txt'])

        normalize_case = st.sidebar.checkbox('Normalize Case')
        clean_stopwords = st.sidebar.checkbox('Stop Words')
        clean_punctuations = st.sidebar.checkbox('Punctuations')
        clean_emails = st.sidebar.checkbox('Emails')
        clean_special_char = st.sidebar.checkbox('Special Charecters')
        clean_numbers = st.sidebar.checkbox('Numbers')
        clean_urls = st.sidebar.checkbox('Urls')
        clean_bad = st.sidebar.checkbox('Explecites')

        if text_file is not None:

            file_details = {'Filename': text_file.name,"Filesize":text_file.size,"Filetype":text_file. type}
            
            st.write(file_details)
            
            
            raw_text = text_file.read().decode('utf-8')
            
            col1,col2 = st.columns(2)
            
            with col1:
                with st.expander('Original Text'):
                    
                    st.write(raw_text)
            

            with col2:
                with st.expander('Processed Text'):



                    if normalize_case:
                        raw_text = raw_text.lower()


                    if clean_stopwords:
                        raw_text = nfx.remove_stopwords(raw_text)

                    if clean_punctuations:
                        raw_text = nfx.remove_punctuations(raw_text)

                    if clean_emails:
                        raw_text = nfx.remove_emails(raw_text)

                    
                    if clean_special_char:
                        raw_text = nfx.remove_special_characters(raw_text)

                    

                    if clean_numbers:
                        raw_text = nfx.remove_numbers(raw_text)

                    
                    if clean_urls:
                        raw_text = nfx.remove_urls(raw_text)

                    
                    if clean_bad:
                        raw_text = nfx.remove_bad_quotes(raw_text)

                    st.write(raw_text)

                    text_downloader(raw_text)


            with st.expander('Text Annalysis'):
                token_result_df = text_analyzer(raw_text)
                st.dataframe(token_result_df)
                make_dowloadable(token_result_df)




            with st.expander('Plot Word Cloud'):
                plot_wordcloud(raw_text)




            with st.expander('Plot Pos Tags'):
                fig = plt.figure()
                sns.countplot(token_result_df['Pos'])
                plt.xticks(rotation=45)
                st.pyplot(fig)
            
            with st.expander('Entities'):
                render_entities(raw_text)


        
        else:
            st.subheader('About')








if __name__ == '__main__':
    main() 