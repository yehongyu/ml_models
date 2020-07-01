nohup tensorflow_model_server \
  --rest_api_port=8501 \
  --model_name=dnn \
  --model_base_path="/Users/aodandan/data/model/dnn" >server.log 2>&1
