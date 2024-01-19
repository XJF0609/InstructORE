import openai
import json
import time
import threading
from sc import DeepClustering
from classifier import ClassifierNew
from chatore import ChatORE
import yaml

def ChatGPT_test2(str):
    completion = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "You are a helpful assistant, skilling in providing general answers to questions in a prescribed format."},
            {"role": "user", "content": str}
        ]
    )
    return completion.choices[0].message["content"]

api_keys = ""

def call_chatgpt(api_key,start,end):
    with open() as fewrel_1600_with_demo_input:
        fewrel_1600_with_demo_input = json.load(fewrel_1600_with_demo_input)
    results_list=[]
    openai.api_key = api_key
    for i in range(start,end):
        sample=fewrel_1600_with_demo_input[i]
        result={}
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                # model="gpt-4-0613",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": sample["input_with_demo"]}
                ]
            )
            ret=completion.choices[0].message["content"]
            result["input_with_demo"]=sample["input_with_demo"]
            result["input"]=sample["input"]
            result["output"]=ret
            result["relation"]=sample["relation"]
            result["head"]=sample["head"]
            result["tail"]=sample["tail"]
            results_list.append(result)
        except Exception as e:
            ret="error"
            i=i-1
        if (i+1) % 100==0:
            path=""
            with open(path, 'w') as fewrel_val_0_400_concat_output:
                json.dump(results_list,fewrel_val_0_400_concat_output)
        time.sleep(1)


def call_chatgpt_2(api_key,start,end):
    with open() as tacred_1000_concat_input:
        tacred_1000_concat_input = json.load(tacred_1000_concat_input)
    results_list=[]
    openai.api_key = api_key
    for i in range(start,end):
        sample=tacred_1000_concat_input[i]
        result={}
        try:
            completion = openai.ChatCompletion.create(
                # model="gpt-3.5-turbo",
                model="gpt-4-0613",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": sample["input_with_demo"]}
                ]
            )
            ret=completion.choices[0].message["content"]
            result["input"]=sample["input"]
            result["sentence"]=sample["sentence"]
            result["output"]=ret
            result["relation"]=sample["relation"]
            result["head"]=sample["head"]
            result["tail"]=sample["tail"]
            results_list.append(result)
        except Exception as e:
            ret="error"
            i=i-1
        if (i+1) % 50==0:
            path=""
            with open(path, 'w') as tacred_output:
                json.dump(results_list,tacred_output)
        time.sleep(1)


def gpt_data(dataset,config):
    for i in config["N"]:
        if dataset=="fewrel":
            threads = []
            for i in range(4):
                thread = threading.Thread(target=call_chatgpt, args=(api_keys,i*400,(i+1)*400,))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
        elif dataset=="tacred":
            threads = []
            for i in range(4):
                thread = threading.Thread(target=call_chatgpt_2, args=(api_keys,i*250,(i+1)*250,))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()


def run(dataset):
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    gpt_data(dataset,config)
    chatore = ChatORE(config)
    chatore.start()


if __name__ == "__main__":
    run("fewrel")