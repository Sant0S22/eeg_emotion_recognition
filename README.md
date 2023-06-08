# EEG EMOTION RECOGNITION
Il progetto è stato sviluppato da Grazia Margarella e Nicola Pio Santorsa nel contesto del corso di Fondamenti di Visione Artificiale e Biometria. 

Nella cartella deliverables sono presenti lo short paper che descrive l'iter progettuale e una presentazione powerpoint riassuntiva.

La cartella EEG_RECOGNITION è il progetto PyCharm da cui è possibile eseguire il codice del progetto. Come prerequisito c'è la necessità di spostare il dataset SEED-IV nella cartella "Seed_IV Database", ossia quella che contiene le cartelle "_raw_data","_feature_smoot", etc.
Come ordine di esecuzione seguire il seguente:
1. main.py: per estrarre i file .mat raw ed eseguire la pipeline di preprocessing proposta nel progetto.
2. main_seediv.py: per estrarre i file .mat preprocessati, in particolare la sola feature DE con LDS;
3. bi-gru-classifier.py e hist_gradient_boost_classifier.py: per eseguire i modelli rilevanti riportati nello short paper.
4. (FACOLTATIVO) gradient_boost_classifer.py: per il modello non riportato nello short paper.

Nella cartella notebook sono presenti i Jupyter Notebook intermedi. Sono da considerare dei draft intermedi per permettere una ricostruzione del lavoro effettuato.