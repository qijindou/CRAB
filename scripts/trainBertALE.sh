python trainBertALE.py --conf conf/demo_rcv1.json \
                        --output_dir output/RCV1_demo_bert_crab/ \
                        --model Bert \
                        --num_al 6 \
                        --num_epochs 10 \
                        --sampling crab \
                        --n_annote 100 \
                        --empty_label 200 