import json
import re
import random
import numpy as np
from sklearn.metrics import adjusted_rand_score
from collections import Counter
import os


def select_data():
    # 读取JSON数据文件
    with open('') as fewrel_val:
        fewrel_val_data = json.load(fewrel_val)
    with open('') as schema:
        schema_data = json.load(schema)
    data_key = fewrel_val_data.keys()
    random_data = {}
    for key in data_key:
        curr_value = fewrel_val_data[key]
        curr_random_data = random.sample(curr_value, 100)
        random_data[schema_data[key]] = curr_random_data
    with open('') as fewrel_val_1600:
        json.dump(random_data, fewrel_val_1600)



def select_data_sample():
    with open('') as fewrel_val:
        fewrel_val_data = json.load(fewrel_val)
    with open('') as schema:
        schema_data = json.load(schema)
    data_key = fewrel_val_data.keys()
    random_data = {}
    for key in data_key:
        curr_value = fewrel_val_data[key]
        curr_random_data = random.sample(curr_value, 5)
        random_data[schema_data[key]] = curr_random_data
    with open('') as fewrel_val_80:
        json.dump(random_data, fewrel_val_80)

def select_input_sample_60():
    with open('') as fewrel_val_1600_concatInput:
        fewrel_val_1600_concatInput_data = json.load(fewrel_val_1600_concatInput)
    random.shuffle(fewrel_val_1600_concatInput_data)
    random_data = random.sample(fewrel_val_1600_concatInput_data,60);
    with open('') as fewrel_val_60_concatInput:
        json.dump(random_data, fewrel_val_60_concatInput)

def select_input_sample_1600():
    with open('') as fewrel_val_1600_concatOutput_new:
        fewrel_val_1600_concatOutput_new_data = json.load(fewrel_val_1600_concatOutput_new)
    random.shuffle(fewrel_val_1600_concatOutput_new_data)
    with open('') as fewrel_val_1600_concatOutput_new_random:
        json.dump(fewrel_val_1600_concatOutput_new_data, fewrel_val_1600_concatOutput_new_random)


def replace_label():
    with open('') as fewrel_val_1600:
        data = json.load(fewrel_val_1600)
    with open('') as shcema:
        data = json.load(shcema)
    


def cat_token(dataset):
    if dataset=="fewrel":
        with open('') as fewrel_val:
            fewrel_val_1600_data = json.load(fewrel_val)
        with open('') as shcema:
            shchema_data = json.load(shcema)
        data_key = fewrel_val_1600_data.keys()
        
        concat_data = []
        for key in data_key:
            # concat_data_list=[]
            curr_value = fewrel_val_1600_data[key]
            for single_value in curr_value:
                curr_obj={}
                tokens=single_value["tokens"]
                curr_str=""
                for token in tokens:
                    curr_str+=token+" "
                curr_str=curr_str.strip()
                curr_str=curr_str.replace(" .",".")
                curr_str=curr_str.replace(" ,",",")
                curr_str=curr_str.replace(" - ","-")
                # curr_str.replace("\\\" ","\\\"")
                curr_obj["sentence"]=curr_str
                curr_obj["head"]=single_value["h"][0]
                curr_obj["tail"]=single_value["t"][0]
                curr_obj["relation"]=shchema_data[key]
                # concat_data_list.append(curr_obj)
                concat_data.append(curr_obj)
        with open('') as fewrel_val_1600_concat:
            json.dump(concat_data, fewrel_val_1600_concat)
    elif dataset=="tacred":
        tacred_all=[]
        tacred_test=open('')
        tacred_train=open('')
        tacred_val=open('')
        tacred_all.extend(read_line(tacred_train))
        tacred_all.extend(read_line(tacred_val))
        tacred_all.extend(read_line(tacred_test))
        with open("") as tacred_concat:
            json.dump(tacred_all,tacred_concat)
            
        
        
def read_line(dataset):
    tacred_all=[]
    for line in dataset:
        single=json.loads(line)
        if single["relation"]=="NA":
            continue
        curr_obj={}
        tokens=single["token"]
        curr_str=""
        for token in tokens:
            curr_str+=token+" "
        curr_str=curr_str.strip()
        curr_str=curr_str.replace(" .",".")
        curr_str=curr_str.replace(" ,",",")
        curr_str=curr_str.replace(" - ","-")
        # curr_str.replace("\\\" ","\\\"")
        curr_obj["sentence"]=curr_str
        curr_obj["head"]=single["h"]["name"]
        curr_obj["tail"]=single["t"]["name"]
        curr_obj["relation"]=single["relation"]
        # concat_data_list.append(curr_obj)
        tacred_all.append(curr_obj)
    return tacred_all

def select_tacred_1000():
    rel=open("","r")
    keys=list(json.load(rel).keys())
    rel_list=[]
    rel_list.extend(keys[31:32])
    rel_list.extend(keys[33:])
    
    tacred_all=open("","r")
    tacred_all=json.load(tacred_all)
    tacred_ten_list=[[] for _ in range(10)]
    
    for single in tacred_all:
        try:
            index=rel_list.index(single["relation"])
            if len(tacred_ten_list[index])>=100:
                continue
            tacred_ten_list[index].append(single)
        except ValueError:
            continue
    tacred_one_dim=[element for row in tacred_ten_list for element in row]
    tacred_1000=open("","w")
    json.dump(tacred_one_dim,tacred_1000)
    for ele in tacred_ten_list:
        
    



def cat_token2(dataset):
    if dataset=="fewrel":
        with open('') as fewrel_val_1600:
            fewrel_val_1600_data = json.load(fewrel_val_1600)
        data_key = fewrel_val_1600_data.keys()
        
        concat_data = []
        for key in data_key:
            # concat_data_list=[]
            curr_value = fewrel_val_1600_data[key]
            for single_value in curr_value:
                curr_obj={}
                tokens=single_value["tokens"]
                curr_str=""
                for token in tokens:
                    curr_str+=token+" "
                curr_str=curr_str.strip()
                curr_str=curr_str.replace(" .",".")
                curr_str=curr_str.replace(" ,",",")
                curr_str=curr_str.replace(" - ","-")
                curr_obj["sentence"]="Please briefly continue to write below with no more than 1 word.\n\n" + curr_str + "\n\nAccording to the above sentence, the abstract relationship between " + single_value["h"][0] +" and " + single_value["t"][0] +" is "
                curr_obj["head"]=single_value["h"][0]
                curr_obj["tail"]=single_value["t"][0]
                curr_obj["relation"]=key
                # concat_data_list.append(curr_obj)
                concat_data.append(curr_obj)
        with open('') as fewrel_val_1600_concat_input:
            json.dump(concat_data, fewrel_val_1600_concat_input)
    elif dataset=="tacred":
        tacred_1000=open("")
        tacred_1000=json.load(tacred_1000)
        input_all=[]
        for single_value in tacred_1000:
            # input=single_value["sentence"] + "\n\nAccording to the above sentence, what's the relationship between " + single_value["head"] +" and " + single_value["tail"] +". "+"Please strictly write the relationship in the format ["+single_value["head"]+"'s type:"+single_value["tail"]+"'s type] no more than 3 words."

            input="Please briefly continue to write below with no more than 3 words.\n\n" + single_value["sentence"] + "\n\nAccording to the above sentence, the abstract relationship between " + single_value["head"] +" and " + single_value["tail"] +" is "

            single_value["input"]=input
            input_all.append(single_value)
        with open('', 'w') as tacred_1000_concat_input:
            json.dump(input_all, tacred_1000_concat_input)

def cat_token3(dataset):
    if dataset=="fewrel":
        with open('') as fewrel_val_1600:
            fewrel_val_1600_data = json.load(fewrel_val_1600)
        data_key = fewrel_val_1600_data.keys()
        
        concat_data = []
        for key in data_key:
            # concat_data_list=[]
            curr_value = fewrel_val_1600_data[key]
            for single_value in curr_value:
                curr_obj={}
                tokens=single_value["tokens"]
                curr_str=""
                for token in tokens:
                    curr_str+=token+" "
                curr_str=curr_str.strip()
                curr_str=curr_str.replace(" .",".")
                curr_str=curr_str.replace(" ,",",")
                curr_str=curr_str.replace(" - ","-")
                curr_obj["input"]="Please give me the answer briefly in the json format of {entity1: 1 word of the entity1's abstract type, entity2: 1 word of the the entity2's abstract type, relation: 1 word of the abstract relation between entity1 and entity2}.\n\n The sentence is: " + curr_str + "\n\nAccording to the above sentence, what are the abstract entity types for entity1: "+single_value["h"][0]+" and entity2: "+ single_value["t"][0]+"? What is the abstract relation between " + single_value["h"][0] +" and " + single_value["t"][0] +"?"
                curr_obj["head"]=single_value["h"][0]
                curr_obj["tail"]=single_value["t"][0]
                curr_obj["relation"]=key
                # concat_data_list.append(curr_obj)
                concat_data.append(curr_obj)
        with open('', 'w') as fewrel_val_1600_concat_input:
            json.dump(concat_data, fewrel_val_1600_concat_input)
    elif dataset=="tacred":
        tacred_1000=open("","r")
        tacred_1000=json.load(tacred_1000)
        input_all=[]
        for single_value in tacred_1000:
            # input=single_value["sentence"] + "\n\nAccording to the above sentence, what's the relationship between " + single_value["head"] +" and " + single_value["tail"] +". "+"Please strictly write the relationship in the format ["+single_value["head"]+"'s type:"+single_value["tail"]+"'s type] no more than 3 words."

            input="Please give me the answer briefly in the json format of {entity1: 1 word of the the entity1's abstract type, entity2: 1 word of the the entity2's abstract type, relation: 1 word of the abstract relation between entity1 and entity2}.\n\n The sentence is: " + single_value["sentence"] + "\n\nAccording to the above sentence, what are the abstract entity types for entity1: "+single_value["head"]+" and entity2: "+ single_value["tail"]+"? What is the abstract relation between " + single_value["head"] +" and " + single_value["tail"] +"?"

            single_value["input"]=input
            input_all.append(single_value)
        with open('') as tacred_1000_concat_input:
            json.dump(input_all, tacred_1000_concat_input)


def stem():
    new_list=[]
    index=0
    with open('') as fewrel_val_1600_concatOutput:
        fewrel_val_1600_concatOutput = json.load(fewrel_val_1600_concatOutput)
    for single_data in fewrel_val_1600_concatOutput:
        output=single_data["output"].lower()
        output=output.replace(".","").replace("\\","").replace("\"","").replace("'","").split(" ")[0]
        output=output.split("-")[0]
        output=output.split("/")[0]
        single_data["output"]=output
        single_data["id"]=index
        new_list.append(single_data)
        index+=1
    with open('') as fewrel_val_1600_concatOutput_new:
        json.dump(new_list, fewrel_val_1600_concatOutput_new)



def mapping():
    with open('') as cluster_list_all:
        cluster_list_all=json.load(cluster_list_all)
    with open("") as fewrel_val_1600_concatOutput_new_random:
        fewrel_val_1600_concatOutput_new_random_data=json.load(fewrel_val_1600_concatOutput_new_random)
    new_list=[]
    arr_pred = np.zeros(1600)
    arr_true = np.zeros(1600)
    for i in range(1600):
        arr_true[i]=int(i/100)
    for i in range(1600):
        curr_obj=fewrel_val_1600_concatOutput_new_random_data[i]
        cluster_pred=cluster_list_all[i]
        cluster_true=int(curr_obj["id"]/100)
        arr_pred[curr_obj["id"]]=cluster_pred
    arr_true=arr_true.astype(int)
    arr_pred=arr_pred.astype(int)
    with open("") as cluster_list_true:
        json.dump(arr_true.tolist(),cluster_list_true)
    with open("") as cluster_list_pred:
        json.dump(arr_pred.tolist(),cluster_list_pred)
    arr_true=arr_true.tolist()
    arr_pred=arr_pred.tolist()

    arr_pred_split=[]
    single_list=[]
    for i in range(1,1601):
        if(i % 100==0):
            single_list.append(arr_pred[i-1])
            arr_pred_split.append({str(int(i/100-1)):single_list})
            single_list=[]
        else:
            single_list.append(arr_pred[i-1])
    arr_pred_split_count=[]
    for i in range(16):
        curr_pred_list=arr_pred_split[i][str(i)]
        curr_count_obj={}
        for e in curr_pred_list:
            curr_count_obj[str(e)]=curr_count_obj[str(e)]+1
        arr_pred_split_count.append({str(i):curr_count_obj})

    with open("") as cluster_list_pred_split:
        json.dump(arr_pred_split,cluster_list_pred_split)
    with open("") as cluster_list_pred_split_count:
        json.dump(arr_pred_split_count,cluster_list_pred_split_count)

def caculate_ari():
    with open('', 'r') as cluster_list_all_demo:
        cluster_list_all_demo_data=json.load(cluster_list_all_demo)
    with open('') as mapping:
        mapping_list=json.load(mapping)
    arr_true=[]
    for map in mapping_list:
        for _ in range(10):
            arr_true.append(map)
    arr_pred=cluster_list_all_demo_data
    ari_score = adjusted_rand_score(arr_true, arr_pred)

def caculate_ari_ignore():
    with open("") as cluster_list_split_with_demo:
        cluster_list_split_with_demo=json.load(cluster_list_split_with_demo)
    arr_true=[]
    arr_pred=[]
    for element in cluster_list_split_with_demo:
        value=list(element.values())[0]
        count=Counter(value)
        most_common_element=max(count,key=count.get)
        arr_true.extend([most_common_element]*len(value))
        arr_pred.extend(value)
    ari_score = adjusted_rand_score(arr_true, arr_pred)


def random_text():
    with open('') as random:
        random_data=json.load(random)
    random_list=[]
    id=0
    for s in random_data:
        ss={}
        ss["sentence"]=s["input"]
        ss["head"]=s["head"]
        ss["tail"]=s["tail"]
        ss["id"]=id
        id+=1
        random_list.append(ss)
    with open("") as fewrel_val_1600_concat_random:
        json.dump(random_list,fewrel_val_1600_concat_random)
    
def fewrel_k_split():  #对数据进行10折交叉处理，分成十份
    with open("") as sentence_1600:
        sentence_1600=json.load(sentence_1600)
    with open("") as entity_1600:
        entity_1600=json.load(entity_1600)
    with open("") as label_1600:
        label_1600=json.load(label_1600)
    for i in range(10):
        test_sentence_list_all=[]
        train_sentence_list_all=[]
        test_entity_list_all=[]
        train_entity_list_all=[]
        test_label_list_all=[]
        train_label_list_all=[]
        for j in range(16):
            test_sentence_list_all.extend(sentence_1600[j*100:(j+1)*100][i*10:(i+1)*10])
            train_sentence_list_all.extend(sentence_1600[j*100:(j+1)*100][0:i*10])
            train_sentence_list_all.extend(sentence_1600[j*100:(j+1)*100][(i+1)*10:100])

            test_entity_list_all.extend(entity_1600[j*100:(j+1)*100][i*10:(i+1)*10])
            train_entity_list_all.extend(entity_1600[j*100:(j+1)*100][0:i*10])
            train_entity_list_all.extend(entity_1600[j*100:(j+1)*100][(i+1)*10:100])

            test_label_list_all.extend(label_1600[j*100:(j+1)*100][i*10:(i+1)*10])
            train_label_list_all.extend(label_1600[j*100:(j+1)*100][0:i*10])
            train_label_list_all.extend(label_1600[j*100:(j+1)*100][(i+1)*10:100])

        path_prefix=""
        if not os.path.exists(path_prefix):
            os.makedirs(path_prefix)

        with open(path_prefix+"test_sentence.json","w") as test_sentence:
            json.dump(test_sentence_list_all,test_sentence)
        with open(path_prefix+"train_sentence.json","w") as train_sentence:
            json.dump(train_sentence_list_all,train_sentence)

        with open(path_prefix+"test_entity.json","w") as test_entity:
            json.dump(test_entity_list_all,test_entity)
        with open(path_prefix+"train_entity.json","w") as train_entity:
            json.dump(train_entity_list_all,train_entity)

        with open(path_prefix+"test_label.json","w") as test_label:
            json.dump(test_label_list_all,test_label)
        with open(path_prefix+"train_label.json","w") as train_label:
            json.dump(train_label_list_all,train_label)

def fewrel_k_split_label(): 
    with open("") as label_1600:
        label_1600=json.load(label_1600)
    for i in range(10):
        test_label_list_all=[]
        train_label_list_all=[]
        for j in range(16):
            
            test_label_list_all.extend(label_1600[j*100:(j+1)*100][i*10:(i+1)*10])
            train_label_list_all.extend(label_1600[j*100:(j+1)*100][0:i*10])
            train_label_list_all.extend(label_1600[j*100:(j+1)*100][(i+1)*10:100])

        path_prefix=""
        if not os.path.exists(path_prefix):
            os.makedirs(path_prefix)

        with open(path_prefix+"test_label_round2.json","w") as test_label:
            json.dump(test_label_list_all,test_label)
        with open(path_prefix+"train_label_round2.json","w") as train_label:
            json.dump(train_label_list_all,train_label)

def shot_mapping():
    with open("") as cluster_list_all_stem:
        cluster_list_all_stem=json.load(cluster_list_all_stem)
    with open("") as center_shot:
        center_shot=json.load(center_shot)
    with open("") as center_11:
        center_11=json.load(center_11)
    curr_shots_map={}
    curr_cluster_seq=center_11[3] #修改
    for i in range(len(curr_cluster_seq)):
        curr_shots_map[curr_cluster_seq[i]]=center_shot[i]
    with open("") as shot_mapping:
        json.dump(curr_shots_map, shot_mapping)

def select_demo():
    with open("") as prediction_2_2:
        prediction_2_2=json.load(prediction_2_2)
    with open("") as cluster_list_all_stem:
        cluster_list_all_stem=json.load(cluster_list_all_stem)
    with open("") as shot_mapping:
        shot_mapping=json.load(shot_mapping)
    with open("") as center_11:
        center_11=json.load(center_11)
    with open("") as fewrel_val_1600_concat_input:
        fewrel_val_1600_concat_input=json.load(fewrel_val_1600_concat_input)
    curr_shots_map={}
    curr_cluster_seq=center_11[3] 
    demo_list=[]
    for i in range(len(curr_cluster_seq)):
        for j in range(100):
            curr_list=shot_mapping[str(curr_cluster_seq[i])]
            demo_with_question=""
            digit_list=prediction_2_2[i*100+j]
            count=0
            right_index=curr_cluster_seq[i]
            index=curr_cluster_seq[i]
            for k in range(len(digit_list)):
                if digit_list[k]==1:
                    count+=1
                    if k!=right_index:
                        index=k
            if count>1:
                other_list=shot_mapping[str(index)]
                for m in range(3):
                    demo_with_question+=curr_list[m]["input"]+curr_list[m]["relation"]+"."
                    demo_with_question+="\n\n"
                    demo_with_question+=other_list[m]["input"]+other_list[m]["relation"]+"."
                    demo_with_question+="\n\n"
                demo_with_question+=fewrel_val_1600_concat_input[i*100+j]["sentence"]
            else:
                for m in range(6):
                    demo_with_question+=curr_list[m]["input"]+curr_list[m]["relation"]+"."
                    demo_with_question+="\n\n"
                demo_with_question+=fewrel_val_1600_concat_input[i*100+j]["sentence"]
            single_input=fewrel_val_1600_concat_input[i*100+j]
            single_input["demo_sentence"]=demo_with_question
            demo_list.append(single_input)
    with open("") as fewrel_1600_with_demo_input:
        json.dump(demo_list, fewrel_1600_with_demo_input)


def select_tacred_demo():
    demo_list=[]
    center_shot=open("")
    center_shot=json.load(center_shot)
    tacred_1000_input=open("")
    tacred_1000_input=json.load(tacred_1000_input)
    for i in range(10):
        center_shot_list_curr=center_shot[i]
        for j in range(100):
            input_with_demo=""
            center_shot_list_half=random.sample(center_shot_list_curr,6)
            for k in range(6):
                input_with_demo+=center_shot_list_half[k]["input"]
                input_with_demo+=" The answer is "+center_shot_list_half[k]["output"]
                input_with_demo+="\n\n"
            input_with_demo+=tacred_1000_input[i*100+j]["input"]
            single_input=tacred_1000_input[i*100+j]
            single_input["input_with_demo"]=input_with_demo
            demo_list.append(single_input)
    with open("") as tacred_1000_concat_input_with_demo:
        json.dump(demo_list, tacred_1000_concat_input_with_demo)


def test_ari():
    arr_true=[]
    arr_pred=[]
    for i in range(16):
        curr=[i for _ in range(100)]
        arr_true.extend(curr)
    for i in range(11):
        curr=[i for _ in range(100)]
        arr_pred.extend(curr)
    arr_pred.extend([1 for _ in range(100)])
    arr_pred.extend([2 for _ in range(100)])
    arr_pred.extend([3 for _ in range(100)])
    arr_pred.extend([4 for _ in range(100)])
    arr_pred.extend([5 for _ in range(100)])
    ari_score = adjusted_rand_score(arr_true, arr_pred)

def select_tuple(dataset):
    if dataset=="tacred":
        center_tuple=open("")
        center_tuple=json.load(center_tuple)[0]
        tacred_1000_with_demo=open("")
        tacred_1000_with_demo=json.load(tacred_1000_with_demo)

        case_list=[]
        for i in range(10):
            curr_tuple=json.loads(center_tuple[i])
            for j in range(100):
                curr_output=tacred_1000_with_demo[i*100+j]
                curr_output.pop("input","")
                curr_output_text=curr_output["output"].replace("The answer is ","").replace(".","")
                
                try:
                    curr_output_tuple=json.loads(curr_output_text)
                    if curr_tuple==curr_output_tuple:
                        continue
                    curr_output["output"]=curr_output_tuple
                    case_list.append(curr_output)
                except:
                    
                    curr_output["output"]=curr_output_text
                    case_list.append(curr_output)
                    continue
        with open("") as tacred_case_with_demo:
            json.dump(case_list, tacred_case_with_demo)
    elif dataset=="fewrel":
        center_relation=open("")
        center_relation=json.load(center_relation)
        fewrel_val_output_with_demo=open("")
        fewrel_val_output_with_demo=json.load(fewrel_val_output_with_demo)
        fewrel_val_output_entity_type=open("")
        fewrel_val_output_entity_type=json.load(fewrel_val_output_entity_type)
        fewrel_1600_concat=open("")
        fewrel_1600_concat=json.load(fewrel_1600_concat)
        case_list=[]
        for i in range(16):
            curr_center_relation=center_relation[1][i]
            for j in range(100):
                curr_case={}
                curr_output_relation=fewrel_val_output_with_demo[i*100+j]["output"].replace(".","").lower()
                if curr_output_relation==curr_center_relation:
                    continue
                curr_output_tuple=json.loads(fewrel_val_output_entity_type[i*100+j]["output"])
                curr_output_tuple["relation"]=curr_output_relation
                curr_case["sentence"]=fewrel_1600_concat[i*100+j]["sentence"]
                curr_case["output"]=curr_output_tuple
                curr_case["relation"]=fewrel_1600_concat[i*100+j]["relation"]
                curr_case["head"]=fewrel_1600_concat[i*100+j]["head"]
                curr_case["tail"]=fewrel_1600_concat[i*100+j]["tail"]
                case_list.append(curr_case)
        with open("") as fewrel_case_with_demo:
            json.dump(case_list, fewrel_case_with_demo)