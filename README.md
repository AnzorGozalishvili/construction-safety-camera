# Current Progress on Drunk Detection Model (see related [paper](https://www.intechopen.com/chapters/58738?fbclid=IwAR120nzkqdIO9eZ2brhIY6LnwabwlyU7S6JzmW_8JBJgvz9dBxa5N61_hUE))
1. Thermal Camera **TOPDON TC001**. (camera specs [here](https://www.amazon.com/Thermal-Android-TOPDON-256x192-Resolution/dp/B0B7LMB22Q))
2. Collecting drunk/sober samples. 
3. Evaluate [existing model](https://github.com/NSEvent/drunk-detection-CNN) performance using this thermal camera and our dataset samples.
4. Retrain new model on combined dataset.

# References
## Video Stream Processing in APP
https://github.com/whitphx/streamlit-webrtc?ref=blog.streamlit.io
https://blog.streamlit.io/how-to-build-the-streamlit-webrtc-component/

## Only Helmet Detection Model Training
[training notebook](notebooks/helmet-detection-model-training-kaggle.ipynb)

## SODA Dataset
https://linjiarui.net/en/portfolio/2022-02-22-SODA-site-object-detection-dataset-for-deep-learning-in-construction
https://scut-scet-academic.oss-cn-guangzhou.aliyuncs.com/SODA/2022.2/VOCv1.zip

## Construction Site Safety Dataset (15 categories)
https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow/data
https://www.kaggle.com/code/hinepo/yolov8-finetuning-for-ppe-detection/output?select=yolov8n.pt
