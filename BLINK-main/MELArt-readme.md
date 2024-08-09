###Train
python3 blink/biencoder/train_biencoder.py --data_path data/artel2/blink_format --output_path models/biencoder --learning_rate 3e-05 --num_train_epochs 3 --max_context_length 128 --max_cand_length 128 --train_batch_size 32 --eval_batch_size 32 --bert_model bert-large-uncased --type_optimization all_encoder_layers --data_parallel --print_interval 100 --eval_interval 2000
###Evaluate
python3 blink/biencoder/eval_biencoder.py  --path_to_model models/artel/biencoder/pytorch_model.bin  --data_path data/artel2/blink_format  --output_path models/artel --encode_batch_size 8 --eval_batch_size 1 --top_k 64 --save_topk_result --bert_model bert-large-uncased --mode test --zeshel False --data_parallel
