#!/bin/bash

# git clone git@github.com:alirezabayatmk/tagger-generator.git
# cd tagger-generator
# chmod +x custom_scripts/tag_setup.sh
# replace when using LUIS cluster
# conda create -y -n tag python=3.7
# conda activate tag
# run this script from the root of the project

echo "creating data directory..."
mkdir data

echo "Starting TAG environment setup..."

#############################################################################################################
#                                         installing required packages                                      #
#############################################################################################################
echo "Installing ipykernel..."
conda install -y ipykernel

# echo "Installing data-prep requirements.txt..."
# pip install -r tag-and-generate-data-prep/requirements.txt

# echo "Installing train requirements.txt..."
# pip install -r tag-and-generate-train/requirements.txt

echo "Installing general requirements.txt..."
pip install -r requirements.txt

# echo "manually installing en-core-web-sm..."
# pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz --no-deps


#############################################################################################################
#                                         running the data preparation                                      #
#############################################################################################################
echo "creating the Inappropriateness.tsv file from raw data..."
python custom_scripts/app_data_preparation.py


#############################################################################################################
#                                         running the data preparation                                      #
#############################################################################################################
echo "Running data preparation..."
python tag-and-generate-data-prep/src/run.py --data_pth data/Inappropriateness.tsv --outpath data/ --style_0_label 'app' --style_1_label 'inapp' --is_unimodal True


#############################################################################################################
#                                        running tokenizer preparation                                      #
#############################################################################################################
echo "Running tokenizer preparation..."
source tag-and-generate-train/scripts/prepare_bpe.sh tagged data
source tag-and-generate-train/scripts/prepare_bpe.sh generated data


#############################################################################################################
#                                                 running tagger                                            #
#############################################################################################################
echo "Running tagger..."
source tag-and-generate-train/scripts/train_tagger.sh tagged appropriatness data
source tag-and-generate-train/scripts/train_generator.sh generated appropriatness data


#############################################################################################################
#                                              running inference                                            #
#############################################################################################################
echo "Running inference..."
source tag-and-generate-train/scripts/inference.sh custom_scripts/input.txt sample tagged generated appropriatness 'app' 'inapp' data 0