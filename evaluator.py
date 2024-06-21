import json
from evaluate import load

def load_json_to_list(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data

if __name__ == "__main__":
    json_file_path = 'dialogue_history.json'
    data_list = load_json_to_list(json_file_path)
    
    predictions = []
    references = []
    
    
    for item in data_list:

        predictions.append(item['answer'])
        references.append(item['ground_truth'])
        print(f"Question: {item['question']}")
        print(f"Answer: {item['answer']}")
        print(f"Ground Truth: {item['ground_truth']}\n")
        bleu = load("bleu")
        rouge = load("rouge")
        results_bleu = bleu.compute(predictions=predictions, references=references)
        print("BLEU Scores:")
        print(results_bleu)
        results_rogue = rouge.compute(predictions=predictions, references=references)
        print("ROUGE Scores:")
        print(results_rogue)
        
    






    

    

    