The src directory is divided into three sections:

classification:
The classification directory contains all the code related to the machine learning models and classification.
It contains a folder 'datasets', in which the labeled data is stored as well as the min and max values of each feature
to normalize the data before predictions can be made. extract_features.py contains the extraction of all the features.
classifier.py forms the base of the sklearn models, containing all methods about the model generation, the optimization,
normalization, etc. The neural network model is contained in neuralnetwork.py. It was outsourced since it doesn't use
sklearn. In both classifier.py and neuralnetwork.py the main function only builds the best models that were estimated,
the optimization of the models were commented out.
evaluation.py contains all the plotting tasks.

text_extraction:
This directory contains everything related to data set generation/extraction and file management.
In order to extract the content of pdf files some methods were implemented in read_pdf.py. Another helper module is
csv_manager.py which contains methods for importing and exporting data from/to csv files. The core is contained in
extract_texts.py. In that file the different text corpora are extracted and converted to dictionaries, which are then
stored in csv files. manipulate_training_data.py was added later to work on already existing files e.g. to make all
words to lemmas or to extend existing files with more data.

user_interface:
This directory contains the graphical user interface. It can be run by executing main.py.