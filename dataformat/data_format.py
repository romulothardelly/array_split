import re
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from termcolor import colored

def clear_string(s):
    s=s.upper()
    #remove all special characters
    s = re.sub(r'[^A-Za-zÀ-ÿ ]+', '', s)
    return s

def split_description(input,output):
    inputs = []
    categories = []
    for index,item in enumerate(input):
        if " " in item:
            a=[]
            a.append(item)
            a.extend(item.split())
            for i in range(len(a)):
                categories.append(output[index])          
       
            inputs.extend(a)
        
        else:
            inputs.append(item)
            categories.append(output[index])
        
    i=0
    while i < len(inputs):
        if len(inputs[i]) < 3:
            inputs.pop(i)
            categories.pop(i)
        else:
            i += 1
    #print('Inputs(',len(inputs),')\n',inputs)
    #print('Categories(',len(categories),')\n',categories)
    return inputs,categories

def minimize_data(inputs, outputs):
    # Dictionary to store grouped inputs and outputs
    grouped_data = {}

    # Loop through inputs and outputs and group them
    for inp, out in zip(inputs, outputs):
        if inp not in grouped_data:
            grouped_data[inp] = set()  # Use a set to avoid duplicate categories
        grouped_data[inp].update(out)  # Add categories to the set

    # Convert the grouped data back to lists
    grouped_inputs = list(grouped_data.keys())
    grouped_outputs = [list(categories) for categories in grouped_data.values()]

    #print("Grouped Inputs(",len(grouped_inputs),"):\n", grouped_inputs)
    #print("Grouped Outputs(",len(grouped_outputs),"):\n", grouped_outputs)

    return grouped_inputs, grouped_outputs

def  prepare_input_output(data):
    input =[clear_string(item["input"]) for item in data]
    output = [item["output"] for item in data]
    inputs,categories = split_description(input,output)


    return minimize_data(inputs,categories)

def create_charset(input_texts):
    all_texts = ''.join(input_texts)
    chars = sorted(set(all_texts))
    char_to_index = {char: idx for idx, char in enumerate(chars)}
    vocab_size = len(chars)
    return char_to_index,vocab_size

def encode_text(text,char_to_index):
    return [char_to_index[char] for char in text]

def prepare_data(data):
    #Split the input and output
    inputs,categoriesx = prepare_input_output(data)
    #Create a character set for input values
    char_to_index,vocab_size=create_charset(inputs)
    #Encode the input text using the loaded char_to_index mapping
    encoded_input = [encode_text(text,char_to_index) for text in inputs]
    #Pad the sequences to the same length
    max_seq_len = max(len(seq) for seq in encoded_input)
    input_sequences = pad_sequences(encoded_input, maxlen=max_seq_len, padding='post')
    #print('Input Sequences(',len(input_sequences),')\n',input_sequences)

    # Step 7: Encode Output Labels
    mlb = MultiLabelBinarizer()
    categories = mlb.fit_transform(categoriesx)
    num_classes = len(mlb.classes_)
    #print('Output Labels(',len(categories),')\n',categories)

    return inputs,categoriesx,char_to_index,vocab_size,max_seq_len,input_sequences,categories,num_classes,mlb

def compare_arrays(arr1, arr2,input_label="",output_label="",verbose=0):
    # Ensure the arrays have the same shape
    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have the same shape to compare")

    # Initialize error count
    errors = 0

    # Compare each element
    for idx, value in np.ndenumerate(arr1):
        if arr1[idx] != arr2[idx]:
            errors += 1
            if verbose:
                print(
                colored(input_label[idx], 'red'),
                colored(output_label[idx], 'red'),
                colored(arr2[idx], 'red'),
                colored(arr1[idx], 'red')
            ) 

    # Calculate total number of elements
    length = arr1.size  # Total number of elements in the array

    # Calculate error ratio
    accuracy = 1-errors / length

    return errors, accuracy