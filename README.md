# CSREC390

## Overview
This GitHub repo is for the review process only. The full dataset and data generation process will be released upon acceptance. 

## Running
### Environment setting
```
conda env create -f CSRec_env.yaml
conda activate csrec
```

### training and testing CSRec
* Step 1: pretrain a sequential recommender system on observational data. 

Notice that training at this step does not require the recommendation and user decision data under intervention. They are only used for evaluation purposes.
```
python run_rec.py --model_type [model type] --data_dir [data directory] --obs_data_name [observational dataset name]\
 --rec_data_name [recommendation dataset name] --dec_data_name [user's decision dataset] --lr [learning rate] \
--num_attention_heads 1 --train_name [saved log name] --epochs [epochs]
```


* Step 2: train CSRec on interventional data with the given pretrained model
```
python run_rec.py --model_type [pretrained model type] --data_dir [data directory] \
--obs_data_name [observational dataset name] --rec_data_name [recommendation dataset name] \
 --dec_data_name [user's decision dataset] --lr [learning rate] --num_attention_heads 1 \
--load_pretrain_model [name for the pretrained model] --train_name [saved log name] --epochs [epochs]
```

All models after training will be saved in the folder ```src/output```.

To test the CSRec after training, add ```do_eval``` boolean argument and provide the model name without .pt. 

Example on GPT4 books dataset 
```
python main.py --model_type [pretrained model type] --data_dir [data directory] \
--obs_data_name [observational dataset name] --rec_data_name [recommendation dataset name] \
 --dec_data_name [user's decision dataset] --lr [learning rate] --num_attention_heads 1 \
--load_pretrain_model [name for the pretrained model] --load_model [name for CSRec model] --do_eval
```

### Train and test other baselines
other baselines can be trained and tested by changing the ```model_type```argument to the following ones:

 *  BSARec Caser, GRU4Rec, SASRec, BERT4Rec, FMLPRec

Example of training SASRec on GPT4 books dataset:
```
python run_rec.py --model_type SASRec --obs_data_name [observational dataset name] \
 --rec_data_name [recommendation dataset name] --dec_data_name [user's decision dataset] --lr [learning rate] \
--num_attention_heads 1 --train_name [saved log name] --epochs [epochs]
```
Testing the model
```
python run_rec.py --model_type SASRec --obs_data_name [observational dataset name] \
 --rec_data_name [recommendation dataset name] --dec_data_name [user's decision dataset] \
--num_attention_heads 1 --load_model [model name] --do_eval
```


### Acknowledgement
This repository is based on [BSARec](https://github.com/yehjin-shin/BSARec/tree/main).































