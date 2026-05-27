#!/usr/bin/env bash

# Define dataset mapping
declare -A dataset_mapping=(
    ["carp"]="carp_boneRGBa_sags_class7"
    ["chameleon"]="chameleonRGBa_tf_class2"
    ["mantle"]="mantleRGBa_tf_class5"
    ["supernova"]="supernovaRGBa_tf_class3"
)

# Get dataset argument
dataset=${1:-"carp"}
mapped_dataset=${dataset_mapping[$dataset]}

if [ -z "$mapped_dataset" ]; then
    echo "Error: Invalid dataset name '$dataset'. Available options: ${!dataset_mapping[@]}"
    exit 1
fi

# Define paths
root_path="./output/$mapped_dataset"
image_path="./ImgData/$mapped_dataset"

# Load API keys from .env file in project root
set -a; source "$(dirname "$0")/../.env"; set +a

api_keys_json="{\"openai_audio\":\"$OPENAI_API_KEY\", \"gpt-4o\":\"$OPENAI_API_KEY\", \"deepseek-chat\":\"$DEEPSEEK_API_KEY\", \"llama3.2-90b-vision\":\"$LLAMA_API_KEY\"}"

model_name="gpt-4o"
embedding_name="image_embedding_entropy_plus_text.npy"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

cd ..

python NLI.py \
    -so "$root_path" \
    --image_path "$image_path" \
    --api_key "$api_keys_json" \
    --llm_name "$model_name" \
    --embedding_name "$embedding_name" \
