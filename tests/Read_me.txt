Hi There,

My name is Himanshu. I am a full time Data Enthusiast and growing my knowlegde 
in Data Science and Machine Learning.

I build an app that could help generate synthetic data for analysis and training
purposes. You can utilize different integrated models that I used in this app
for data generation. Below is the list of models - 

1. Openhermes (Local) - is an LLM model from huggingface which is optimized to 
give structural outputs. I setup is this in my local machine to utilize my 
machine's capabilities. This is the best option free option that you could use 
to generate data. But there are limitation to this model becuase it runs on
local machine. It is suggested to use other models if you want to generate rows
more than 1000 because of local machine limitations.
Input - No. of Rows, Schema JSON

2. Openrouter API (cloud) - [add details for this model]
Input - No. of Rows, Schema JSON

3. CTGAN (faker python library) - [add details for this model]
Input - No. of Rows, Schema JSON, Example rows>=2 in JSON (opetional)


But I also created a config that you can use to changes the configurations on 
the app. such as,
- Model used in local machine
- Model used for openrouter API
- max_tokens
- row_nums limit
- batch size (in case of big row counts)

You can utilize this app to get domain specific synthetic data.