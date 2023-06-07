import pandas as pd
from nltk.tokenize import RegexpTokenizer
import nltk as nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

# federalist txt
authors = 'data/federalist.txt'

final_dataset = pd.DataFrame(columns=['author', 'body'])

f = open(authors)
full_text = f.read()
f.close

# Clean linebreaks and encodings
full_text_lower = full_text.replace(u'\ufeff', '')
full_text_lower = full_text.replace(u'\n', ' ')
full_text_lower = full_text.lower()

# Remove punctuation
tokenizer = RegexpTokenizer(r'\w+')
full_text = tokenizer.tokenize(full_text_lower)
preprocessed_text = ' '.join(full_text)

# exctact author, chapter title and text

startindex = []
endindex = []
authors = []
titles = []

for i in range(0, 85):  # Split into the 85 chapters
    search = 'federalist no ' + str(i + 1)
    startindex.append(preprocessed_text.find(search))  # Start point of chapter
    endindex.append(startindex[i] + len(search) + 1)  # End point of chapter

# find author of each chapter
for i in range(84):
    find_jay = preprocessed_text.find('jay', endindex[i], startindex[i + 1])
    find_hamilton = preprocessed_text.find('hamilton', endindex[i], startindex[i + 1])
    find_madison = preprocessed_text.find('madison', endindex[i], startindex[i + 1])
    find_HwM = preprocessed_text.find('madison with hamilton', endindex[i], startindex[i + 1])
    if find_jay != -1:
        df = {
            'author': 'jay',
            'body': preprocessed_text[find_jay + 4:startindex[i + 1]]
        }
    elif find_HwM != -1:
        df = {
            'author': 'madison with hamilton',
            'body': preprocessed_text[find_madison + 22:startindex[i + 1]]
        }
    elif find_hamilton != -1:
        df = {
            'author': 'hamilton',
            'body': preprocessed_text[find_hamilton + 9:startindex[i + 1]]
        }
    elif find_madison != -1:
        df = {
            'author': 'madison',
            'body': preprocessed_text[find_madison + 8:startindex[i + 1]]
        }
    final_dataset = final_dataset.append(df, ignore_index=True)

find_hamilton = preprocessed_text.find('hamilton', endindex[84])
df = {
    'author': 'hamilton',
    'body': preprocessed_text[find_hamilton + 9:1124005]
}
final_dataset = final_dataset.append(df, ignore_index=True)

# open each row in df and apply preproccessing

for i, text in enumerate(final_dataset['body']):
    # remove stop words
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_output = [word for word in words if word not in stop_words]
    # # # stemming
    # stemmer = PorterStemmer()
    # stemmed_output = [stemmer.stem(word) for word in filtered_output]
    # # # Join the stemmed words back into a single string
    new_text = ' '.join(filtered_output)

    final_dataset.at[i, 'body'] = new_text
#
#     #
#     #
#     #
final_dataset.to_csv("data/text_stop.csv", index=False)
