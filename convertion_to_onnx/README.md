# How to convert the model into ONNX

To run inference in C++, the `.ckpt` file may need to be converted into an `.onnx` file.  
This can be done by following these steps:

0. **Prerequisite**: ensure that **Apptainer** is installed.  
1. **Pull the container image**:  
   `singularity pull docker://justdrew/onnxconversion`
2. **Run the container**:  
   `apptainer shell onnxconversion_latest.sif`  
   > **Note:** Make sure the container has access to both the conversion script  
   > (`Tracking_DC/scripts/onnx_conversion.py`) and the `.ckpt` file.  
3. **Run the conversion script**:  
   ```bash
   python scripts/onnx_conversion.py \
       --weightpath <PATH_TO_CKPT_FILE> \
       --outputpath <PATH_WHERE_TO_SAVE_THE_ONNX_FILE>
   ```