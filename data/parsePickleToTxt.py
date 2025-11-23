import pickle
import os
import vocab

def main(pickle_file_path, text_file_path):
    # Läs in pickled data från filen
    with open(pickle_file_path, 'rb') as file:
        data = pickle.load(file)
    
    # Konvertera datat till sträng (om det inte redan är en sträng)
    if not isinstance(data, str):
        data = str(data)
    
    # Skriv strängen till en textfil
    with open(text_file_path, 'w') as file:
        file.write(data)

if __name__ == "__main__":

    pickle_file_path = '/Users/kevinkokalari/Documents/KEX/MHCH-DAMI-main/data/clothing/vocab.pkl'
    text_file_path = '/Users/kevinkokalari/Documents/KEX/MHCH-DAMI-main/data/clothing/vocab.txt'

    main(pickle_file_path, text_file_path)