# Tv-Script-Generation
Generate TV script for Simpsons using Recurrent Neural Networks(RNNs)

<img src="./udacity.PNG" alt="Udacity" width="250px"/>

---

## Overview

In this project, we will generate a TV script using a recurrent neural network. We will train the network on existing scripts then use it to generate an original piece of writing. 

## Dataset

We'll be using a Seinfeld dataset of scripts from 9 seasons. 

The data can be found in ./data/Seinfeld_Scripts.txt

**Dataset Stats**

Roughly the number of unique words: 46367

Number of lines: 109233

Average number of words in each line: 5.544240293684143

The lines 0 to 10: (Data Sample)

jerry: do you know what this is all about? do you know, why were here? to be out, this is out...and out is one of the single most enjoyable experiences of life. people...did you ever hear people talking about we should go out? this is what theyre talking about...this whole thing, were all out now, no one is home. not one person here is home, were all out! there are people trying to find us, they dont know where we are. (on an imaginary phone) did you ring?, i cant find him. where did he go? he didnt tell me where he was going. he must have gone out. you wanna go out you get ready, you pick out the clothes, right? you take the shower, you get all ready, get the cash, get your friends, the car, the spot, the reservation...then youre standing around, what do you do? you go we gotta be getting back. once youre out, you wanna get back! you wanna go to sleep, you wanna get up, you wanna go out again tomorrow, right? where ever you are in life, its my feeling, youve gotta go. 

jerry: (pointing at georges shirt) see, to me, that button is in the worst possible spot. the second button literally makes or breaks the shirt, look at it. its too high! its in no-mans-land. you look like you live with your mother. 

george: are you through? 

jerry: you do of course try on, when you buy? 

george: yes, it was purple, i liked it, i dont actually recall considering the buttons.

## Data Pre-processing

**Lookup Table**

To create a word embedding, you first need to transform the words to ids. In this function, create two dictionaries:

- Dictionary to go from the words to an id, we'll call vocab_to_int
- Dictionary to go from the id to word, we'll call int_to_vocab

**Tokenize Punctuation**

We'll be splitting the script into a word array using spaces as delimiters. However, punctuations like periods and exclamation marks can create multiple ids for the same word. For example, "bye" and "bye!" would generate two different word ids.

Implement the function token_lookup to return a dict that will be used to tokenize symbols like "!" into "||Exclamation_Mark||". Create a dictionary for the following symbols where the symbol is the key and value is the token:

- Period ( . )
- Comma ( , )
- Quotation Mark ( " )
- Semicolon ( ; )
- Exclamation mark ( ! )
- Question mark ( ? )
- Left Parentheses ( ( )
- Right Parentheses ( ) )
- Dash ( - )
- Return ( \n )

This dictionary will be used to tokenize the symbols and add the delimiter (space) around it. This separates each symbols as its own word, making it easier for the neural network to predict the next word. Make sure you don't use a value that could be confused as a word; for example, instead of using the value "dash", try using something like "||dash||".

**Batching Data**

Batch_data function to batch words data into chunks of size batch_size using the TensorDataset and DataLoader classes.

You can batch words using the DataLoader, but it will be up to you to create feature_tensors and target_tensors of the correct size and content for a given sequence_length.

For example, say we have these as input:

words = [1, 2, 3, 4, 5, 6, 7]
sequence_length = 4

Your first feature_tensor should contain the values:

[1, 2, 3, 4]

And the corresponding target_tensor should just be the next "word"/tokenized word value:

5

This should continue with the second feature_tensor, target_tensor being:

[2, 3, 4, 5]  # features
6             # target

## Neural Network

The network consits of 3 layers making the Recurrent Neural Network

- Layer-1: Embedding Layer 
- Layer-2: LSTM layer
- Layer-3: Fully connected linear layer

Optimization Function: Adam
Loss Function: CrossEntropyLoss

Choice of hyperparameters:

**Data Parameters**
| Parameter | Value |
| --- | ----------- |
| batch_size | 128 |
| num_epochs | 15 |
| learning_rate | 0.001 |
| sequence_length | 8 |

**Model Parameters**
| Parameter | Value |
| --- | ----------- |
| embedding_dim | 200 |
| hidden_dim | 200 |
| n_layers | 2 |
| show_every_n_batches | 1000 |

Starting point of my hyper parameters was based on examples shown through the sentiment analysis example. My thought process for each is as described below:

- sequence_length: This is a tricky parameter as you want it to be the average length of sentances you are trying to generate. If chosen too small then it does poorly on longer sentances. Larger sequence length does result in longer training time. I found any number between 6-8 would be a good choice.
- batch_size: Batch size controls the accuracy of the estimate of the error gradient when training neural networks. So a batch size of 128 as chosen in this case means 128 samples from the training dataset will be used to estimate the error gradient before the model weights are updated. Larger batch size may make it difficult to train due to memory constraints on GPU. While, smaller batch sizes are noisy, offering a regularizing effect and lower generalization error.
- num_epochs: One training epoch means that the learning algorithm has made one pass through the training dataset. Hence a choice of 15 meant the model would go through the entired dataset 15 times, where examples were separated into randomly selected batch size groups. 15 was adequate to reach a loss value below 3.5. usually you want it to be large enough such that your loss is no longer decreasing.
- learning_rate: Learning rate controls how quickly or slowly a neural network model learns a problem i.e. the amount that the weights are updated during training. A learning rate that is too large will result in weight updates that will be too large and the performance of the model (such as its loss on the training dataset) will oscillate over training epochs. this is exactly what I observed with a learning rate of 0.01, reducing it to 0.001 made the model converge.
- embedding_dim: Embedding is an alternative to one-hot-encoding. So, instead of ending up with huge one-hot encoded vectors we can use an embedding matrix to keep the size of each vector much smaller. The vectors of each embedding get updated while training the neural network. This also allows us to visualize relationships between words. To create and embedding we must specify 3 arguments. input_dim: This is the size of the vocabulary in the text data. output_dim: This is the size of the vector space in which words will be embedded. (the parameter being set here) and input_length: This is the length of input sequences. Usually larger the size, more information can be extracted from the data. The final number comes through experimentation.
- hidden_dim: This is the feature detection layer and once again larger the size more features can be extracted from the data.
- n_layers: The numbe of layers can be selected based on the complexity of problem begin solved. In this case I thought 2 was enough to get a loss value below 3.5. But for better performing model maybe adding more depth/layers would help.

Reference:

- https://machinelearningmastery.com/how-to-control-the-speed-and-stability-of-training-neural-networks-with-gradient-descent-batch-size/
- https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/
- https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

## RESULTS

Script generated by the model:

jerry: thou- five percent.

elaine: so what happened? you want to be a good person?

elaine: no, i don't know how it would be good.

jerry: i was in a little nervous about a snack, 000.

george:(to kramer) i can't stand anywhere.

elaine: no, no, no no no no no no no. i'm sorry about this guy.

jerry: i can't find this.

jerry: no, no. no, i don't know. you know, i think i'm not married.

jerry:(sarcastic) no no no no, no, no, no, no no.

kramer: hey! you want to go, we'll get rid of that network.

george: i don't want you to get it out.

newman:(to elaine) what is that?

jerry:(sarcastic) you can't handle it!

newman: no, no no no. i can't believe it was a little problem, but i was just going in handicapped manner of 1990.

jerry: you don't want to be a little religious.

george: i can't believe it. you know, i don't know if they were in mortal p- chat.

george: so what are you doing here?

jerry: well, it's not a lot hilly.

jerry: i think i don't want to get a little adjustment.

hoyt: so, what are you doing?

george: i don't know. it's a little more.

elaine: oh, no, i got to talk about the tractor.

jerry: you know, i can't do that.

george: i think i could do this.

kramer: no, no, no, i didn't think so.

kramer: well.

jerry: i don't think so.

elaine: no, it's a misprint, it's a long time.

george: no no no no. no! no, no.

man: so what




