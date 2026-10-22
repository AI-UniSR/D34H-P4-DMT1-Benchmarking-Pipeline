#!/bin/bash
 
# Azure ML Job Create Script
 
SUBSCRIPTION_ID="ab211f7b-463f-4833-9605-d260e596a35a"
RESOURCE_GROUP="16f50b7b-b9b7-1d4b-0c62-3a11cf1ad1ee"
WORKSPACE_NAME="16f50b7bb9b71d4b0c623a11cf1ad1ee"
YAML_FILE="/home/azureuser/cloudfiles/code/Users/arslan.muhammad/UseCase-Code/data-science/benchmarking-pipeline/benchmarking_pipeline.yml"

echo "Starting Azure ML job creation..."
 
az ml job create \
  --subscription "$SUBSCRIPTION_ID" \
  --resource-group "$RESOURCE_GROUP" \
  --workspace-name "$WORKSPACE_NAME" \
  --file "$YAML_FILE" \
  --stream
 
echo "Azure ML job creation command finished."