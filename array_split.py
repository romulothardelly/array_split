
from dataformat.data_format import *
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, Input
from keras.optimizers import Adam
from keras.callbacks import Callback
import numpy as np
from termcolor import colored

#define data"""
"""
data=[{"input":"ROUPA DE CAMA","output":["CASA"]},
      {"input":"ROUPA DE FESTA","output":["CASA"]},
      {"input":"ROUPA DE TRABALHO","output":["TRABALHO"]},
      {"input":"COMPUTADOR PESSOAL","output":["CASA"]},
      {"input":"COMPUTADOR DO TRABALHO","output":["TRABALHO"]},
      {"input":"SALA DE REUNIÃO","output":["TRABALHO"]},
      {"input":"SALA DE JANTA","output":["CASA"]},
      {"input":"CARRO PESSOAL","output":["CASA"]},
      {"input":"CARRO EMERGÊNCIA","output":["CASA","TRABALHO"]},
      {"input":"ESCADA DE FIBRA ISOLADO","output":["TRABALHO"]},
      {"input":"CADEIRA DE ESCRITÓRIO3","output":["CASA","TRABALHO"]},
      {"input":"SAPATO DO TRABALHO","output":["TRABALHO"]},
      {"input":"SAPATO DE CASA","output":["CASA"]},
      {"input":"SAPATO DE CAMPO","output":["TRABALHO"]},]
"""
data = [
    {"input": "ROUPA DE CAMA", "output": ["CASA"]},
    {"input": "ROUPA DE FESTA", "output": ["CASA"]},
    {"input": "ROUPA DE TRABALHO", "output": ["TRABALHO"]},
    {"input": "COMPUTADOR PESSOAL", "output": ["CASA"]},
    {"input": "COMPUTADOR DO TRABALHO", "output": ["TRABALHO"]},
    {"input": "SALA DE REUNIÃO", "output": ["TRABALHO"]},
    {"input": "SALA DE JANTA", "output": ["CASA"]},
    {"input": "CARRO PESSOAL", "output": ["CASA"]},
    {"input": "CARRO EMERGÊNCIA", "output": ["CASA", "TRABALHO"]},
    {"input": "ESCADA DE FIBRA ISOLADO", "output": ["TRABALHO"]},
    {"input": "CADEIRA DE ESCRITÓRIO", "output": ["CASA", "TRABALHO"]},
    {"input": "SAPATO DO TRABALHO", "output": ["TRABALHO"]},
    {"input": "SAPATO DE CASA", "output": ["CASA"]},
    {"input": "SAPATO DE CAMPO", "output": ["TRABALHO"]},
    {"input": "MESA DE JANTAR", "output": ["CASA"]},
    {"input": "MESA DE REUNIÃO", "output": ["TRABALHO"]},
    {"input": "GAVETEIRO DE ESCRITÓRIO", "output": ["TRABALHO"]},
    {"input": "ESTANTE DA SALA", "output": ["CASA"]},
    {"input": "ESTANTE DO ESCRITÓRIO", "output": ["TRABALHO"]},
    {"input": "COPO DE VIDRO", "output": ["CASA"]},
    {"input": "COPO TÉRMICO", "output": ["TRABALHO"]},
    {"input": "PRATELEIRA DE LIVROS", "output": ["CASA"]},
    {"input": "PRATELEIRA DO ESCRITÓRIO", "output": ["TRABALHO"]},
    {"input": "SOFÁ DA SALA", "output": ["CASA"]},
    {"input": "CADEIRA DE TRABALHO", "output": ["TRABALHO"]},
    {"input": "LIVRO DE ESTUDO", "output": ["TRABALHO"]},
    {"input": "LIVRO DE CASA", "output": ["CASA"]},
    {"input": "CANETA DE ESCRITÓRIO", "output": ["TRABALHO"]},
    {"input": "CANETA DECORATIVA", "output": ["CASA"]},
    {"input": "CARTEIRA PESSOAL", "output": ["CASA"]},
    {"input": "DOCUMENTOS DO TRABALHO", "output": ["TRABALHO"]},
    {"input": "DOCUMENTOS PESSOAIS", "output": ["CASA"]},
    {"input": "MOCHILA DO TRABALHO", "output": ["TRABALHO"]},
    {"input": "MOCHILA DE VIAGEM", "output": ["CASA"]},
    {"input": "TELEVISÃO DA SALA", "output": ["CASA"]},
    {"input": "COMPUTADOR DO ESCRITÓRIO", "output": ["TRABALHO"]},
    {"input": "CAFETEIRA DA COZINHA", "output": ["CASA"]},
    {"input": "CAFETEIRA DO ESCRITÓRIO", "output": ["TRABALHO"]},
    {"input": "TAPETE DA SALA", "output": ["CASA"]},
    {"input": "TAPETE DO ESCRITÓRIO", "output": ["TRABALHO"]},
    {"input": "CAMISETA DE CASA", "output": ["CASA"]},
    {"input": "CAMISETA DE TRABALHO", "output": ["TRABALHO"]},
    {"input": "VENTILADOR DA SALA", "output": ["CASA"]},
    {"input": "VENTILADOR DO ESCRITÓRIO", "output": ["TRABALHO"]},
    {"input": "AR CONDICIONADO DA SALA", "output": ["CASA"]},
    {"input": "AR CONDICIONADO DO ESCRITÓRIO", "output": ["TRABALHO"]},
    {"input": "ABAJUR DO QUARTO", "output": ["CASA"]},
    {"input": "LUMINÁRIA DO ESCRITÓRIO", "output": ["TRABALHO"]},
    {"input": "APONTADOR DO ESCRITÓRIO", "output": ["TRABALHO"]},
    {"input": "CALCULADORA FINANCEIRA", "output": ["TRABALHO"]},
    {"input": "PORTA-RETRATO", "output": ["CASA"]},
    {"input": "PORTA-RETRATO DO ESCRITÓRIO", "output": ["TRABALHO"]},
    {"input": "AGENDA PESSOAL", "output": ["CASA"]},
    {"input": "AGENDA DO TRABALHO", "output": ["TRABALHO"]},
    {"input": "CADERNO DE ESTUDOS", "output": ["TRABALHO"]},
    {"input": "CADERNO DE CASA", "output": ["CASA"]},
    {"input": "QUADRO DECORATIVO", "output": ["CASA"]},
    {"input": "QUADRO DE PLANEJAMENTO", "output": ["TRABALHO"]},
    {"input": "PAPEL DE IMPRESSÃO", "output": ["TRABALHO"]},
    {"input": "PAPEL DE PRESENTE", "output": ["CASA"]},
    {"input": "BOLSA DE VIAGEM", "output": ["CASA"]},
    {"input": "BOLSA DO TRABALHO", "output": ["TRABALHO"]},
    {"input": "CHAVE DE CASA", "output": ["CASA"]},
    {"input": "CHAVE DO TRABALHO", "output": ["TRABALHO"]},
    {"input": "RELÓGIO DE PAREDE", "output": ["CASA"]},
    {"input": "RELÓGIO DE MESA", "output": ["TRABALHO"]},
    {"input": "GARRAFA DE ÁGUA", "output": ["CASA"]},
    {"input": "GARRAFA TÉRMICA", "output": ["TRABALHO"]},
    {"input": "TRAVESSEIRO", "output": ["CASA"]},
    {"input": "MOUSE DO COMPUTADOR", "output": ["TRABALHO"]},
    {"input": "CADERNETA PESSOAL", "output": ["CASA"]},
    {"input": "ESFEROGRÁFICA AZUL", "output": ["TRABALHO"]},
    {"input": "PASTA DE ARQUIVOS", "output": ["TRABALHO"]},
    {"input": "PASTA DE DOCUMENTOS PESSOAIS", "output": ["CASA"]},
    {"input": "BALANÇA DE COZINHA", "output": ["CASA"]},
    {"input": "FERRAMENTA DE MONTAGEM", "output": ["TRABALHO"]},
    {"input": "GRAMPEADOR", "output": ["TRABALHO"]},
    {"input": "LÁPIS DE ESCOLA", "output": ["CASA"]},
    {"input": "ESTOJO DE CANETAS", "output": ["TRABALHO"]},
    {"input": "PAINEL DE AVISOS", "output": ["TRABALHO"]},
    {"input": "FRIGOBAR DO ESCRITÓRIO", "output": ["TRABALHO"]},
    {"input": "COBRELEITO", "output": ["CASA"]},
    {"input": "LIXEIRA DE COZINHA", "output": ["CASA"]},
    {"input": "LIXEIRA DO ESCRITÓRIO", "output": ["TRABALHO"]},
    {"input": "EXTINTOR DE INCÊNDIO", "output": ["CASA", "TRABALHO"]},
    {"input": "FONE DE OUVIDO", "output": ["TRABALHO"]},
    {"input": "FONE DE OUVIDO PESSOAL", "output": ["CASA"]},
    {"input": "TECLADO DO COMPUTADOR", "output": ["TRABALHO"]},
    {"input": "CAPA DE CHUVA", "output": ["CASA"]}]

 
inputs,categoriesx,char_to_index,vocab_size,max_seq_len,input_sequences,categories,num_classes,mlb = prepare_data(data)

print('Input Sequences(',len(input_sequences),')\n',input_sequences)
print('Categories:(',len(categories),')\n',categories)
print('Type of categories:',type(categories))

# Define the LSTM model
model = Sequential([
    Input(shape=(max_seq_len,)),  # Input shape is the padded sequence length
    Embedding(input_dim=vocab_size, output_dim=64, input_length=max_seq_len),
    LSTM(10, return_sequences=True),  # Return the sequence to sequence
     LSTM(10, return_sequences=False),          
    Dense(num_classes, activation='sigmoid')  # Predict the next character from the vocabulary
])

model.compile(optimizer=Adam(learning_rate=0.03), loss='binary_crossentropy', metrics=['accuracy'])

class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):

        predictions = model.predict(input_sequences,verbose=0)
        #print('Predicted Labels\n',predictions)
        predicted_labels = (predictions > 0.5).astype(int)
        #print('Predicted Labels\n',predicted_labels)

        errors,accuracy=compare_arrays(categories,predicted_labels)
        print("Epoch: ",epoch,", Accuracy:", accuracy,". Errors:", errors) 
        if accuracy ==1.0:
            print(f"Stopping training at epoch {epoch + 1} because accuracy reached 1.0")
            self.model.stop_training = True  # Stops training

accuracy_callback = MyCallback()

model.fit(input_sequences, categories, epochs=1000,verbose=0,callbacks=[accuracy_callback])

predictions = model.predict(input_sequences,verbose=0)
predicted_labels = (predictions > 0.5).astype(int)

errors,accuracy=compare_arrays(categories,predicted_labels)
print("Accuracy:", accuracy,". Errors:", errors) 







