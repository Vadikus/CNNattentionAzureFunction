#!/bin/bash

# python3 -m venv .venv
# source .venv/bin/activate

# func init --python
# func new --name funcpytest1 --template "HTTP trigger"
# 
# func start
# az login
# az group create --name AzureFucntionsQuickstart-rg --location westus2
# az storage account create --name functionsquickstart1 --location westus2 --resource-group AzureFucntionsQuickstart-rg --sku Standard_LRS
# az functionapp create --resource-group AzureFucntionsQuickstart-rg --os-type Linux --consumption-plan-location westus2 --runtime python --runtime-version 3.7 --functions-version 2 --name funcvadtest1 --storage-account functionsquickstart1
# func azure functionapp publish funcvadtest1



# Function app and storage account names must be unique.
export AZURE_STORAGE_ACCOUNT=kerasstorage
export ResourceGroup=CVchallenge
functionAppName=attentioncv
region=westus2
pythonVersion=3.7 #3.6 also supported
shareName=kerasfileshare
directoryName=mydir
shareId=kerasshareid
mountPath=/home/.keras

# Create a resource group.
az group create --name $ResourceGroup --location $region

# Create an Azure storage account in the resource group.
az storage account create \
  --name $AZURE_STORAGE_ACCOUNT \
  --location $region \
  --resource-group $ResourceGroup \
  --sku Standard_LRS

# Set the storage account key as an environment variable. 
export AZURE_STORAGE_KEY=$(az storage account keys list -g $ResourceGroup -n $AZURE_STORAGE_ACCOUNT --query '[0].value' -o tsv)

# Create a serverless function app in the resource group.
az functionapp create \
  --name $functionAppName \
  --storage-account $AZURE_STORAGE_ACCOUNT \
  --consumption-plan-location $region \
  --resource-group $ResourceGroup \
  --os-type Linux \
  --runtime python \
  --runtime-version $pythonVersion \
  --functions-version 3

# Work with Storage account using the set env variables.
# Create a share in Azure Files.
az storage share create \
  --name $shareName 

# Create a directory in the share.
az storage directory create \
  --share-name $shareName \
  --name $directoryName

az webapp config storage-account add \
  --resource-group $ResourceGroup \
  --name $functionAppName \
  --custom-id $shareId \
  --storage-type AzureFiles \
  --share-name $shareName \
  --account-name $AZURE_STORAGE_ACCOUNT \
  --mount-path $mountPath \
  --access-key $AZURE_STORAGE_KEY

az webapp config storage-account list \
  --resource-group $ResourceGroup \
  --name $functionAppName

