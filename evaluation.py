import json
import re
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
from rouge_score import rouge_scorer
import code_bert_score
from codebleu import calc_codebleu


def clean_result(text):
    pattern = r'```javascript\s*\n(.*?)(?:\n```|$)'
    
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).rstrip()
    return None


if __name__ == "__main__":
    ## Uncomment for first execution
    # nltk.download('punkt')
    # nltk.download('wordnet')
    # nltk.download('punkt_tab')
    
    with open("data\\data.json", "r", encoding="utf-8") as f:
        applets = json.load(f)
    
    n = len(applets)
    
    model1_metrics = {}
    model1_metrics["bleu"] = 0
    model1_metrics["rougeL"] = 0
    model1_metrics["rouge1"] = 0
    model1_metrics["rouge2"] = 0
    model1_metrics["meteor"] = 0
    model1_metrics["codebertscore"] = 0
    model1_metrics["codebleu"] = 0
    
    
    model2_metrics = {}
    model2_metrics["bleu"] = 0
    model2_metrics["rougeL"] = 0
    model2_metrics["rouge1"] = 0
    model2_metrics["rouge2"] = 0
    model2_metrics["meteor"] = 0
    model2_metrics["codebertscore"] = 0
    model2_metrics["codebleu"] = 0
    

    i=1
    for applet in applets.values():
        print(f"Applet {i}")
        
        test_code = applet["filter_code"]
        model1_res = clean_result(applet["deepseek-ai/deepseek-coder-1.3b-instruct"])
        model2_res = clean_result(applet["Qwen/Qwen2.5-Coder-1.5B-Instruct"])
    
    
        # BLEU
        model1_bleu = sentence_bleu(references=[test_code], hypothesis=model1_res)
        model2_bleu = sentence_bleu(references=[test_code], hypothesis=model2_res)
        
        model1_metrics["bleu"] += model1_bleu
        model2_metrics["bleu"] += model2_bleu
        
        
        # ROUGE-L, ROUGE-1, ROUGE-2
        scorer = rouge_scorer.RougeScorer(["rougeL", "rouge1", "rouge2"])
        model1_rougeL = scorer.score(target=test_code, prediction=model1_res)["rougeL"].fmeasure
        model1_rouge1 = scorer.score(target=test_code, prediction=model1_res)["rouge1"].fmeasure
        model1_rouge2 = scorer.score(target=test_code, prediction=model1_res)["rouge2"].fmeasure
        
        model2_rougeL = scorer.score(target=test_code, prediction=model2_res)["rougeL"].fmeasure
        model2_rouge1 = scorer.score(target=test_code, prediction=model2_res)["rouge1"].fmeasure
        model2_rouge2 = scorer.score(target=test_code, prediction=model2_res)["rouge2"].fmeasure
        
        model1_metrics["rougeL"] += model1_rougeL
        model1_metrics["rouge1"] += model1_rouge1
        model1_metrics["rouge2"] += model1_rouge2
        
        model2_metrics["rougeL"] += model2_rougeL
        model2_metrics["rouge1"] += model2_rouge1
        model2_metrics["rouge2"] += model2_rouge2

                
        # METEOR        
        model1_meteor = meteor_score(references=[word_tokenize(test_code)], hypothesis=word_tokenize(model1_res))
        model2_meteor = meteor_score(references=[word_tokenize(test_code)], hypothesis=word_tokenize(model2_res))

        model1_metrics["meteor"] += model1_meteor
        model2_metrics["meteor"] += model2_meteor
        
        
        # CodeBERTScore
        model1_CBS = float(code_bert_score.score(refs=[test_code], cands=[model1_res], lang='javascript')[2][0])
        model2_CBS = float(code_bert_score.score(refs=[test_code], cands=[model2_res], lang='javascript')[2][0])
                
        model1_metrics["codebertscore"] += model1_CBS
        model2_metrics["codebertscore"] += model2_CBS
        
        
        # CodeBLEU
        model1_CB = calc_codebleu([test_code], [model1_res], "javascript")["codebleu"]
        model2_CB = calc_codebleu([test_code], [model2_res], "javascript")["codebleu"]
        
        model1_metrics["codebleu"] += model1_CB
        model2_metrics["codebleu"] += model2_CB
        
        i+=1

    # Calculate average
    model1_metrics["bleu"] /= n
    model1_metrics["rougeL"] /= n
    model1_metrics["rouge1"] /= n
    model1_metrics["rouge2"] /= n
    model1_metrics["meteor"] /= n
    model1_metrics["codebertscore"] /= n
    model1_metrics["codebleu"] /= n
    
    model2_metrics["bleu"] /= n
    model2_metrics["rougeL"] /= n
    model2_metrics["rouge1"] /= n
    model2_metrics["rouge2"] /= n
    model2_metrics["meteor"] /= n
    model2_metrics["codebertscore"] /= n
    model2_metrics["codebleu"] /= n


    metrics = {}
    metrics["model1"] = model1_metrics
    metrics["model2"] = model2_metrics
    
    with open("data\\metrics.json", "w") as f:
        json.dump(metrics, f, indent=3, separators=(',', ': '))
        