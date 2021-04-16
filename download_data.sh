#!/usr/bin/env bash

readonly DATA_DIR="$(pwd)"/data
readonly USERNAME="$1"
readonly DATASET_NAME="$2"
readonly DATASET_REF="${USERNAME}/${DATASET_NAME}"
readonly ZIPPED_FILENAME="${DATASET_NAME}.zip"
readonly KAGGLE_JSON="$HOME/.kaggle/kaggle.json"

if [ ! -f "$KAGGLE_JSON" ] ; then
    echo -e "\e[1;31m ERROR: kaggle.json do not exist. \e[0m"
else
    # Make sure kaggle api is latest version
    echo "Upgrading kaggle api."
    pip install kaggle --upgrade

    ## Download and Unzip the data
    # Download
    kaggle datasets download "$DATASET_REF"

    # Unzip
    if [ ! -d "$DATA_DIR" ] ; then
       echo "Creating data directory." 
       mkdir "$DATA_DIR"
    fi

    unzip "$ZIPPED_FILENAME" -d "$DATA_DIR"

    # Delete zipfile
    rm "$ZIPPED_FILENAME"
fi


