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
    pattern = r'```(?:javascript|js)\n(.*?)```'
    
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        return '\n\n'.join(matches)
    else:
        return text


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
    
    
    # model3_metrics = {}
    # model3_metrics["bleu"] = 0
    # model3_metrics["rougeL"] = 0
    # model3_metrics["rouge1"] = 0
    # model3_metrics["rouge2"] = 0
    # model3_metrics["meteor"] = 0
    # model3_metrics["codebertscore"] = 0
    # model3_metrics["codebleu"] = 0
    
    model1_name = "qwen2.5-coder-7b-instruct"
    model2_name = "yi-coder-9b-chat"
    # model3_name = "deepseek-coder-6.7b-instruct"

    i=1
    for applet in applets.values():
        print(f"Applet {i}")
        
        test_code = applet["filter_code"]
        model1_res = clean_result(applet[model1_name])
        model2_res = clean_result(applet[model2_name])
        # model3_res = clean_result(applet["model3_name"])
    
        
        # BLEU
        model1_bleu = sentence_bleu(references=[test_code], hypothesis=model1_res)
        model2_bleu = sentence_bleu(references=[test_code], hypothesis=model2_res)
        # model3_bleu = sentence_bleu(references=[test_code], hypothesis=model3_res)
        
        model1_metrics["bleu"] += model1_bleu
        model2_metrics["bleu"] += model2_bleu
        # model3_metrics["bleu"] += model3_bleu
        
        
        # ROUGE-L, ROUGE-1, ROUGE-2
        scorer = rouge_scorer.RougeScorer(["rougeL", "rouge1", "rouge2"])
        model1_rougeL = scorer.score(target=test_code, prediction=model1_res)["rougeL"].fmeasure
        model1_rouge1 = scorer.score(target=test_code, prediction=model1_res)["rouge1"].fmeasure
        model1_rouge2 = scorer.score(target=test_code, prediction=model1_res)["rouge2"].fmeasure
        
        model2_rougeL = scorer.score(target=test_code, prediction=model2_res)["rougeL"].fmeasure
        model2_rouge1 = scorer.score(target=test_code, prediction=model2_res)["rouge1"].fmeasure
        model2_rouge2 = scorer.score(target=test_code, prediction=model2_res)["rouge2"].fmeasure
        
        # model3_rougeL = scorer.score(target=test_code, prediction=model3_res)["rougeL"].fmeasure
        # model3_rouge1 = scorer.score(target=test_code, prediction=model3_res)["rouge1"].fmeasure
        # model3_rouge2 = scorer.score(target=test_code, prediction=model3_res)["rouge2"].fmeasure
        
        model1_metrics["rougeL"] += model1_rougeL
        model1_metrics["rouge1"] += model1_rouge1
        model1_metrics["rouge2"] += model1_rouge2
        
        model2_metrics["rougeL"] += model2_rougeL
        model2_metrics["rouge1"] += model2_rouge1
        model2_metrics["rouge2"] += model2_rouge2
        
        # model3_metrics["rougeL"] += model3_rougeL
        # model3_metrics["rouge1"] += model3_rouge1
        # model3_metrics["rouge2"] += model3_rouge2

                
        # METEOR        
        model1_meteor = meteor_score(references=[word_tokenize(test_code)], hypothesis=word_tokenize(model1_res))
        model2_meteor = meteor_score(references=[word_tokenize(test_code)], hypothesis=word_tokenize(model2_res))
        # model3_meteor = meteor_score(references=[word_tokenize(test_code)], hypothesis=word_tokenize(model3_res))

        model1_metrics["meteor"] += model1_meteor
        model2_metrics["meteor"] += model2_meteor
        # model3_metrics["meteor"] += model3_meteor
        
        
        # CodeBERTScore
        model1_CBS = float(code_bert_score.score(refs=[test_code], cands=[model1_res], lang='javascript')[2][0])
        model2_CBS = float(code_bert_score.score(refs=[test_code], cands=[model2_res], lang='javascript')[2][0])
        # model3_CBS = float(code_bert_score.score(refs=[test_code], cands=[model3_res], lang='javascript')[2][0])

        model1_metrics["codebertscore"] += model1_CBS
        model2_metrics["codebertscore"] += model2_CBS
        # model3_metrics["codebertscore"] += model3_CBS
        
        
        # CodeBLEU
        model1_CB = calc_codebleu([test_code], [model1_res], "javascript")["codebleu"]
        model2_CB = calc_codebleu([test_code], [model2_res], "javascript")["codebleu"]
        # model3_CB = calc_codebleu([test_code], [model3_res], "javascript")["codebleu"]
        
        model1_metrics["codebleu"] += model1_CB
        model2_metrics["codebleu"] += model2_CB
        # model3_metrics["codebleu"] += model3_CB
        
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
    
    # model3_metrics["bleu"] /= n
    # model3_metrics["rougeL"] /= n
    # model3_metrics["rouge1"] /= n
    # model3_metrics["rouge2"] /= n
    # model3_metrics["meteor"] /= n
    # model3_metrics["codebertscore"] /= n
    # model3_metrics["codebleu"] /= n


    metrics = {}
    metrics[model1_name] = model1_metrics
    metrics[model2_name] = model2_metrics
    # metrics[model3_name] = model3_metrics
    
    with open("data\\metrics.json", "w") as f:
        json.dump(metrics, f, indent=3, separators=(',', ': '))
        