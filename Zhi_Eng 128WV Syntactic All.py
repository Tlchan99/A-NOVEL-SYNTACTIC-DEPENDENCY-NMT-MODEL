import codecs, sys, pydot, os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pickle import load
from numpy import array, newaxis, argmax
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Reshape, Dense, LSTM, Bidirectional, Input, Concatenate, TimeDistributed, Dropout, Embedding, RepeatVector
from keras.callbacks import ModelCheckpoint
from nltk.translate.bleu_score import corpus_bleu
from attention_decoder import AttentionDecoder

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))
 
# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
 
# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)
 
# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X
 
# one hot encode target sequence99
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y
 
# define NMT model
def define_model(src_text_vocab, tar_text_vocab, src_timesteps, tar_timesteps, src_POS_vocab_size, tar_POS_vocab_size, src_head_vocab_size, tar_head_vocab_size, src_arch_vocab_size, tar_arch_vocab_size, n_units):

	input_head = Input(shape=(src_timesteps,), name = 'Source_Head')
	head_one_hot = Embedding(src_head_vocab_size, n_units, input_length=src_timesteps, mask_zero=True, name="Source_head_Embedding")(input_head)
	head_one_hot = Bidirectional(LSTM(n_units, dropout=0.3))(head_one_hot)
    
	input_arch = Input(shape=(src_timesteps,), name = 'Source_Arch')
	arch_one_hot = Embedding(src_arch_vocab_size, n_units, input_length=src_timesteps, mask_zero=True, name="Source_arch_Embedding")(input_arch)
	arch_one_hot = Bidirectional(LSTM(n_units, dropout=0.3))(arch_one_hot)
    
	input_POS = Input(shape=(src_timesteps,), name="Source_POS")
	POS_one_hot = Embedding(src_POS_vocab_size, n_units, input_length=src_timesteps, mask_zero=True, name="Source_POS_Embedding")(input_POS)
	POS_one_hot = Bidirectional(LSTM(n_units, dropout=0.3))(POS_one_hot)
    
	input_text = Input(shape=(src_timesteps,), name="Source_Text")
	text_one_hot = Embedding(src_text_vocab, n_units, input_length=src_timesteps, mask_zero=True, name="Source_Text_Embedding")(input_text)
	text_one_hot = Bidirectional(LSTM(n_units, dropout=0.3))(text_one_hot)
    
	encoder_output = Concatenate()([text_one_hot, head_one_hot, arch_one_hot, POS_one_hot])
	decoder_input = RepeatVector(tar_timesteps)(encoder_output)
    
	decoder_output_head = AttentionDecoder(n_units,tar_timesteps, tar_head_vocab_size, src_timesteps, name="Target_Head")(decoder_input)
	head_prediction = Bidirectional(LSTM(n_units, dropout=0.3))(decoder_output_head)
	decoder_output_arch = AttentionDecoder(n_units,tar_timesteps, tar_arch_vocab_size, src_timesteps, name="Target_Arch")(decoder_input)
	arch_prediction = Bidirectional(LSTM(n_units, dropout=0.3))(decoder_output_arch)
	decoder_output_POS = AttentionDecoder(n_units,tar_timesteps, tar_POS_vocab_size, src_timesteps, name="Target_POS")(decoder_input)
	POS_one_hot_prediction = Bidirectional(LSTM(n_units, dropout=0.3))(decoder_output_POS)
	decoder_output_text = AttentionDecoder(n_units,tar_timesteps, tar_text_vocab, src_timesteps, name="Target_Text")(decoder_input)
	text_prediction = Bidirectional(LSTM(n_units, dropout=0.3))(decoder_output_text)

	decoder_prediction = Concatenate()([text_one_hot, text_prediction, head_prediction, arch_prediction, POS_one_hot_prediction])
	AD_input = RepeatVector(tar_timesteps)(decoder_prediction)
	final_decoder_output_text = AttentionDecoder(n_units,tar_timesteps, tar_text_vocab, src_timesteps, name="Final_Target_Text")(AD_input)

	model = Model(inputs = [input_text, input_head, input_arch, input_POS], outputs = [final_decoder_output_text, decoder_output_text, decoder_output_head, decoder_output_arch, decoder_output_POS], name="Autoencoder")
	return model

# load datasets
dataset = load_clean_sentences('Eng_Zhi+All-both rev25.pkl')
train = load_clean_sentences('Eng_Zhi+All-train rev25.pkl')
test = load_clean_sentences('Eng_Zhi+All-test rev25.pkl')

# prepare english tokenizer (dummy for Chinese)
zhi_tokenizer = create_tokenizer(dataset[:, 0])
zhi_vocab_size = len(zhi_tokenizer.word_index) + 1
zhi_length = max_length(dataset[:, 0])
print('Chinese Vocabulary Size: %d' % zhi_vocab_size)
print('Chinese Max Length: %d' % zhi_length)

zhi_head_tokenizer = create_tokenizer(dataset[:, 2])
zhi_head_vocab_size = len(zhi_head_tokenizer.word_index) + 1
zhi_head_length = max_length(dataset[:, 2])
print('Chinese Head Vocabulary Size: %d' % zhi_head_vocab_size)
print('Chinese Head Max Length: %d' % zhi_head_length)

zhi_arch_tokenizer = create_tokenizer(dataset[:, 3])
zhi_arch_vocab_size = len(zhi_arch_tokenizer.word_index) +1
zhi_arch_length = max_length(dataset[:, 3])
print('Chinese Arch Vocabulary Size: %d' % zhi_arch_vocab_size)
print('Chinese Arch Max Length: %d' % zhi_arch_length)

zhi_POS_tokenizer = create_tokenizer(dataset[:, 4])
zhi_POS_vocab_size = len(zhi_POS_tokenizer.word_index) + 1
zhi_POS_length = max_length(dataset[:, 4])
print('zhi_POS Vocabulary Size: %d' % zhi_POS_vocab_size)
print('zhi_POS Max Length: %d' % zhi_POS_length)

# prepare german tokenizer (dummy for Portuguese)
eng_tokenizer = create_tokenizer(dataset[:, 1])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 1])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % eng_length)

eng_head_tokenizer = create_tokenizer(dataset[:, 6])
eng_head_vocab_size = len(eng_head_tokenizer.word_index) + 1
eng_head_length = max_length(dataset[:, 6])
print('English Head Vocabulary Size: %d' % eng_head_vocab_size)
print('English Head Max Length: %d' % eng_head_length)

eng_arch_tokenizer = create_tokenizer(dataset[:, 7])
eng_arch_vocab_size = len(eng_arch_tokenizer.word_index) + 1
eng_arch_length = max_length(dataset[:, 7])
print('English Arch Vocabulary Size: %d' % eng_arch_vocab_size)
print('English Arch Max Length: %d' % eng_arch_length)

eng_POS_tokenizer = create_tokenizer(dataset[:, 8])
eng_POS_vocab_size = len(eng_POS_tokenizer.word_index) + 1
eng_POS_length = max_length(dataset[:, 8])
print('eng_POS Vocabulary Size: %d' % eng_POS_vocab_size)
print('eng_POS Max Length: %d' % eng_POS_length)

# prepare training data
trainX_text = encode_sequences(eng_tokenizer, eng_length, train[:,1])
print('trainX_text.shape', trainX_text.shape)
print('trainX_text.ndim', trainX_text.ndim)
trainX_head = encode_sequences(eng_head_tokenizer, eng_length, train[:,6])
trainX_arch = encode_sequences(eng_arch_tokenizer, eng_length, train[:,7])
#for i, row in enumerate(trainX_arch):
#	print('trainX',trainX_arch[i], dataset[i,0], dataset[i,1],dataset[i,2],dataset[i,3],dataset[i,4],dataset[i,5],dataset[i,6],dataset[i,7],dataset[i,8])
trainX_POS = encode_sequences(eng_POS_tokenizer, eng_length, train[:,8])
print('trainX_head.shape', trainX_head.shape)
print('trainX_arch.shape', trainX_arch.shape)
print('trainX_POS.shape', trainX_POS.shape)
print('trainX_POS.ndim', trainX_POS.ndim)
trainY_text = encode_sequences(zhi_tokenizer, zhi_length, train[:,0])
trainY_text = encode_output(trainY_text, zhi_vocab_size)
trainY_head = encode_sequences(zhi_head_tokenizer, zhi_length, train[:, 2])
trainY_head = encode_output(trainY_head, zhi_head_vocab_size)
trainY_arch = encode_sequences(zhi_arch_tokenizer, zhi_length, train[:, 3])
#for i, row in enumerate(trainY_arch):
#	print('trainY_arch1',trainY_arch[i], dataset[i,0], dataset[i,1],dataset[i,2],dataset[i,3],dataset[i,4],dataset[i,5],dataset[i,6],dataset[i,7],dataset[i,8])
trainY_arch = encode_output(trainY_arch, zhi_arch_vocab_size)
trainY_POS = encode_sequences(zhi_POS_tokenizer, zhi_length, train[:, 4])
trainY_POS = encode_output(trainY_POS, zhi_POS_vocab_size)
print('trainY_text.shape', trainY_text.shape)
print('trainY_text.ndim', trainY_text.ndim)
print('trainY_head.shape', trainY_head.shape)
print('trainY_arch.shape', trainY_arch.shape)
print('trainY_POS.shape', trainY_POS.shape)
print('trainY_POS.ndim', trainY_POS.ndim)

# prepare validation data
testX_text = encode_sequences(eng_tokenizer, eng_length, test[:, 1])
print('testX_text.shape', testX_text.shape)
print('testX_text.ndim', testX_text.ndim)
testX_head = encode_sequences(eng_head_tokenizer, eng_length, test[:,6])
testX_arch = encode_sequences(eng_arch_tokenizer, eng_length, test[:,7])
testX_POS = encode_sequences(eng_POS_tokenizer, eng_length, test[:,8])
print('testX_head.shape', testX_head.shape)
print('testX_arch.shape', testX_arch.shape)
print('testX_POS.shape', testX_POS.shape)
print('testX_POS.ndim', testX_POS.ndim)
testY_text = encode_sequences(zhi_tokenizer, zhi_length, test[:,0])
testY_text = encode_output(testY_text, zhi_vocab_size)
testY_head = encode_sequences(zhi_head_tokenizer, zhi_length, test[:, 2])
testY_head = encode_output(testY_head, zhi_head_vocab_size)
testY_arch = encode_sequences(zhi_arch_tokenizer, zhi_length, test[:, 3])
testY_arch = encode_output(testY_arch, zhi_arch_vocab_size)
testY_POS = encode_sequences(zhi_POS_tokenizer, zhi_length, test[:, 4])
testY_POS = encode_output(testY_POS, zhi_POS_vocab_size)
print('testY_text.shape', testY_text.shape)
print('testY_text.ndim', testY_text.ndim)
print('testY_head.shape', testY_head.shape)
print('testY_arch.shape', testY_arch.shape)
print('testY_POS.shape', testY_POS.shape)
print('testY_POS.ndim', testY_POS.ndim)

# compile model
model = define_model(eng_vocab_size, zhi_vocab_size, eng_length, zhi_length, eng_POS_vocab_size, zhi_POS_vocab_size, eng_head_vocab_size, zhi_head_vocab_size, eng_arch_vocab_size, zhi_arch_vocab_size, 192)
model.compile(optimizer='adam', loss=['categorical_crossentropy','categorical_crossentropy', 'categorical_crossentropy', 'binary_crossentropy', 'categorical_crossentropy'], loss_weights = [0.5, 0.2, 0.05, 0.05, 0.2], metrics=['acc', 'acc', 'acc', 'acc', 'acc'])
print(model.summary())

# fit model
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=2, save_best_only=True, mode='min')  

history = model.fit([trainX_text, trainX_head, trainX_arch, trainX_POS], [trainY_text, trainY_text, trainY_head, trainY_arch, trainY_POS], 
                    epochs=50, validation_steps = 55, steps_per_epoch = 55,
                    validation_data=([testX_text, testX_head, testX_arch, testX_POS], 
                                     [testY_text, testY_text, testY_head, testY_arch, testY_POS]), callbacks=[checkpoint], verbose=2)
#batch_size=128,
loss_weights = [0.5,0.2, 0.05, 0.05, 0.2]
acc, val_acc = list(), list()
for element in range(0,len(history.history['Final_Target_Text_acc'])):
	acc.append(history.history['Final_Target_Text_acc'][element]*loss_weights[0]+history.history['Target_Text_acc'][element]*loss_weights[1]+history.history['Target_Head_acc'][element]*loss_weights[2]+history.history['Target_Arch_acc'][element]*loss_weights[3]+history.history['Target_POS_acc'][element]*loss_weights[4])
	val_acc.append(history.history['val_Final_Target_Text_acc'][element]*loss_weights[0]+history.history['val_Target_Text_acc'][element]*loss_weights[1]+history.history['val_Target_Head_acc'][element]*loss_weights[2]+history.history['val_Target_Arch_acc'][element]*loss_weights[3]+history.history['val_Target_POS_acc'][element]*loss_weights[4])
print('model.metrics_names', model.metrics_names)

print("loss", history.history['loss'])
print("acc", acc)
print("val_loss", history.history['val_loss'])
print("val_acc", val_acc)
print('Final_Target_Text_acc', history.history['Final_Target_Text_acc'])
print('Target_Text_acc', history.history['Target_Text_acc'])
print('Target_Head_acc', history.history['Target_Head_acc'])
print('Target_Arch_acc', history.history['Target_Arch_acc'])
print('Target_POS_acc', history.history['Target_POS_acc'])

print('Final_Target_Text_loss', history.history['Final_Target_Text_loss'])
print('Target_Text_loss', history.history['Target_Text_loss'])
print('Target_Head_loss', history.history['Target_Head_loss'])
print('Target_Arch_loss', history.history['Target_Arch_loss'])
print('Target_POS_loss', history.history['Target_POS_loss'])

print('val_Final_Target_Text_acc', history.history['val_Final_Target_Text_acc'])
print('val_Target_Text_acc', history.history['val_Target_Text_acc'])
print('val_Target_Head_acc', history.history['val_Target_Head_acc'])
print('val_Target_Arch_acc', history.history['val_Target_Arch_acc'])
print('val_Target_POS_acc', history.history['val_Target_POS_acc'])

print('val_Final_Target_Text_loss', history.history['val_Final_Target_Text_loss'])
print('val_Target_Text_loss', history.history['val_Target_Text_loss'])
print('val_Target_Head_loss', history.history['val_Target_Head_loss'])
print('val_Target_Arch_loss', history.history['val_Target_Arch_loss'])
print('val_Target_POS_loss', history.history['val_Target_POS_loss'])

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
 
# generate target given source sequence
def predict_sequence(model, text_tokenizer, POS_tokenizer, head_tokenizer, arch_tokenizer, sources_text, sources_head, sources_arch, sources_POS):
	sources_text = sources_text.reshape(1, len(sources_text))
	sources_head = sources_head.reshape(1, len(sources_head))
	sources_arch = sources_arch.reshape(1, len(sources_arch))
	sources_POS = sources_POS.reshape(1, len(sources_POS))
	final_prediction_text, prediction_text, prediction_head, prediction_arch, prediction_POS = model.predict([sources_text, sources_head, sources_arch, sources_POS], verbose=2)
	prediction_text = prediction_text.reshape(zhi_length, zhi_vocab_size)
	prediction_head = prediction_head.reshape(zhi_length, zhi_head_vocab_size)
	prediction_arch = prediction_arch.reshape(zhi_length, zhi_arch_vocab_size)
	prediction_POS = prediction_POS.reshape(zhi_length, zhi_POS_vocab_size)
	final_prediction_text = final_prediction_text.reshape(zhi_length, zhi_vocab_size)
#	print('prediction shape', prediction_text.shape, prediction_head.shape, prediction_arch.shape, prediction_POS.shape)
	text_integers = [argmax(vector_text) for vector_text in prediction_text]
	head_integers = [argmax(vector_head) for vector_head in prediction_head]
	arch_integers = [argmax(vector_arch) for vector_arch in prediction_arch]
	POS_integers = [argmax(vector_POS) for vector_POS in prediction_POS]
	final_text_integers = [argmax(vector_final_text) for vector_final_text in final_prediction_text]
	final_target_text, target_text, target_head, target_arch, target_POS = list(), list(), list(), list(), list()

	for i in final_text_integers:
		final_text = word_for_id(i, text_tokenizer)
		if final_text is None:
			break
		final_target_text.append(final_text)
	' '.join(final_target_text)

	for j in text_integers:
		text = word_for_id(j, text_tokenizer)
		if text is None:
			break
		target_text.append(text)
	' '.join(target_text)

	for k in head_integers:
		head = word_for_id(k,head_tokenizer)
		if head is None:
			break
		target_head.append(head)
		' '.join(target_head)

	for l in arch_integers:
		arch = word_for_id(l,arch_tokenizer)
		if arch is None:
			break
		target_arch.append(arch)
		' '.join(target_arch)

	for m in POS_integers:
		POS = word_for_id(m,POS_tokenizer)
		if POS is None:
			break
		target_POS.append(POS)
	' '.join(target_POS)

	return ' '.join(final_target_text),' '.join(target_text),' '.join(target_head),' '.join(target_arch),' '.join(target_POS)
 
# evaluate the skill of the model BLEU-1,2,3,4
def evaluate_model(model, text_tokenizer, POS_tokenizer, head_tokenizer, arch_tokenizer, sources_text, sources_head, sources_arch, sources_POS, raw_dataset):
	actual_text, actual_POS, actual_head, actual_arch, predicted_text, predicted_POS, predicted_head, predicted_arch, final_predicted_text = list(), list(), list(), list(), list(), list(), list(), list(), list()
	i=0
	for (text, head, arch, POS) in zip(sources_text ,sources_head, sources_arch, sources_POS):
		final_translated_text, translated_text, translated_head, translated_arch, translated_POS = predict_sequence(model, text_tokenizer, POS_tokenizer, head_tokenizer, arch_tokenizer, text, head, arch, POS)
		target_text, source_text, source_head, source_arch, source_POS, target_text_split, target_head, target_arch, target_POS = raw_dataset[i]
		if i < 35:
#			print('source_text =[%s], target_text =[%s], source_head=[%s], source_arch =[%s], source_POS =[%s], target_text_split =[%s], target_head =[%s], target_arch =[%s], target_POS=[%s]' % (source_text, target_text, source_head, source_arch, source_POS, target_text_split, target_head, target_arch, target_POS))
			print('source_text =[%s], target_text =[%s], final_predicted_text=[%s]' % (source_text, target_text, final_translated_text))
			print('source_text =[%s], target_text =[%s], predicted_text =[%s]' % (source_text, target_text, translated_text))
			print('source_head =[%s], target_head =[%s], predicted_head =[%s]' % (source_head, target_head, translated_head))
			print('source_arch =[%s], target_arch =[%s], predicted_arch =[%s]' % (source_arch, target_arch, translated_arch))
			print('source_POS =[%s], target_POS =[%s], predicted_POS =[%s]' % (source_POS, target_POS, translated_POS))

		actual_text.append([target_text.split()])
		actual_head.append([target_head.split()])
		actual_arch.append([target_arch.split()])
		actual_POS.append([target_POS.split()])
		final_predicted_text.append(final_translated_text.split())
		predicted_text.append(translated_text.split())
		predicted_head.append(translated_head.split())
		predicted_arch.append(translated_arch.split())
		predicted_POS.append(translated_POS.split())
		i=i+1

	print('final text')
	print('BLEU-1: %f' % corpus_bleu(actual_text, final_predicted_text, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual_text, final_predicted_text, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual_text, final_predicted_text, weights=(0.33, 0.33, 0.33, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual_text, final_predicted_text, weights=(0.25, 0.25, 0.25, 0.25)))
	print('text')
	print('BLEU-1: %f' % corpus_bleu(actual_text, predicted_text, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual_text, predicted_text, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual_text, predicted_text, weights=(0.33, 0.33, 0.33, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual_text, predicted_text, weights=(0.25, 0.25, 0.25, 0.25)))
	print('head')
	print('BLEU-1: %f' % corpus_bleu(actual_head, predicted_head, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual_head, predicted_head, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual_head, predicted_head, weights=(0.33, 0.33, 0.33, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual_head, predicted_head, weights=(0.25, 0.25, 0.25, 0.25)))
	print('arch')
	print('BLEU-1: %f' % corpus_bleu(actual_arch, predicted_arch, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual_arch, predicted_arch, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual_arch, predicted_arch, weights=(0.33, 0.33, 0.33, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual_arch, predicted_arch, weights=(0.25, 0.25, 0.25, 0.25)))
	print('POS')
	print('BLEU-1: %f' % corpus_bleu(actual_POS, predicted_POS, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual_POS, predicted_POS, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual_POS, predicted_POS, weights=(0.33, 0.33, 0.33, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual_POS, predicted_POS, weights=(0.25, 0.25, 0.25, 0.25)))

# test on some training sequences
print('train')
evaluate_model(model, zhi_tokenizer, zhi_POS_tokenizer, zhi_head_tokenizer, zhi_arch_tokenizer, trainX_text, trainX_head, trainX_arch, trainX_POS, train)

# test on some test sequences
print('test')
evaluate_model(model, zhi_tokenizer, zhi_POS_tokenizer, zhi_head_tokenizer, zhi_arch_tokenizer, testX_text, testX_head, testX_arch, testX_POS, test)

# plot model
plot_model(model, to_file='model.png', show_shapes=True)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(acc)
plt.plot(val_acc)

#plt.plot(Target_acc)
#plt.plot(history.history['Target_Text_acc'])
#plt.plot(history.history['Target_Head_acc'])
#plt.plot(history.history['Target_Arch_acc'])
#plt.plot(history.history['Target_POS_acc'])
#plt.plot(history.history['Target_Text_loss'])
#plt.plot(history.history['Target_Head_loss'])
#plt.plot(history.history['Target_Arch_loss'])
#plt.plot(history.history['Target_POS_loss'])
#plt.plot(Target_loss)
#plt.plot(history.history['val_Target_Text_acc'])
#plt.plot(history.history['val_Target_Head_acc'])
#plt.plot(history.history['val_Target_Arch_acc'])
#plt.plot(history.history['val_Target_POS_acc'])
#plt.plot(history.history['val_Target_Text_loss'])
#plt.plot(history.history['val_Target_Head_loss'])
#plt.plot(history.history['val_Target_Arch_loss'])
#plt.plot(history.history['val_Target_POS_loss'])

plt.title('model train vs validation loss/acc')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
