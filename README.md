# Topic Modeling using LDA
Cluster the documents into different topics.

The porject is created using ktrain which is a lightweight wrapper for the deep learning library TensorFlow Keras (and other libraries) to help build, train, and deploy neural networks and other machine learning models.

You can get to know more about ktrain at https://github.com/amaiya/ktrain.

To run the repository follow the steps:

(1) Clone the repository:

      git clone https://github.com/AdityaSadana/Topic-Modeling-using-LDA
      
(2) Install Requirements:

    pip install -r requirements.txt
    
(3) Train The Model:

    python train_model.py

(4) Run the model:

    python get_prediction.py
    
(5) Enter your text. The model would predict the topic and other documents related to it.

The Document Clusters can be visualized as

![Document_Visualization](https://github.com/AdityaSadana/Topic-Modeling-using-LDA/blob/main/Visualization_plot.png)