### Project Reproduction Instructions
This is the code for Deep Multiple Clustering Voting (EDMCV) based on ESM.
EDMCV provides a new unsupervised method for classifying thermophilic proteins represented by thermal stability, which offers a low-cost way to classify the properties of unlabeled sequences, thereby improving the mining of unknown sequence data. 
In addition, mining feature motifs from classified thermostable protein sequences provides important guidance for building a knowledge base and further optimizing protein property designs. 
The efficiency and scalability of this method lay a solid foundation for research into thermophilic proteins and their applications.

### Runtime Environment
- Python version: 3.7.0
- Library versions: See `requirements.txt`

### File Descriptions and Code Execution Instructions
- `esm_encode`: ESM-1v encode
  - `main.py` : Essential file for running the code. Main function.
  - `raw_data_1708` : Protein sequence data
      ```
      python main.py
      ```
      
- `dec_cluster`: deep clustering
  - `train.py` : Essential file for running the code. Training.
  - `load_weight_test.py` : Essential file for running the code. Output model-encoded data.
      ```
      python train.py
      python load_weight_test.py
      ```

- `multi-clustering_voting`: multi-clustering voting
  - `run.sh` : Scripts to run the code.
  - `config.py` : Modify the number of clustering categories.
  - `model.py` : Modify the clustering model.
      ```
      sh run.sh
      ```
