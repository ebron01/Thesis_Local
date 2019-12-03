Hello
if you read this, you are in the right place.

To train Glove on your own corpus, make changes as stated below:


https://stackoverflow.com/questions/48962171/how-to-train-glove-algorithm-on-my-own-corpus

This is how you run the model

$ git clone http://github.com/stanfordnlp/glove
$ cd glove && make
To train it on your own corpus, you just have to make changes to one file, that is demo.sh.

Remove the script from if to fi after 'make'. Replace the CORPUS name with your file name 'corpus.txt' There is another if loop at the end of file 'demo.sh'

if [ "$CORPUS" = 'text8' ]; then
Replace text8 with you file name.  Run the demo.sh once the changes are made.

$ ./demo.sh
Make sure your corpus file is in the correct format.
You'll need to prepare your corpus as a single text file with all words separated by one or more spaces or tabs. If your corpus has multiple documents, the documents (only) should be separated by new line characters.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


AND for second part as stated in src/demo readme you  can see help if you type 

./build/vocab_count

change the parameters as help states. (you can use the tokenizer in StanfordParser file be create a tokenized file )


%%%%%%%%%%%%
vectorloader.py code is made by me based on stackoverflow question. It loads vectors and can be used to get closest word vectors for a given word.

eval/python has some evaluation codes as written like vectorloader.py

glovecreator.sh makes same job as demo.sh